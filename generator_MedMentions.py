import csv, random, torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from collections import defaultdict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LogitsProcessor,
    LogitsProcessorList,
)
from rdflib import Graph, RDF, RDFS
from rdflib.namespace import OWL

'''
TODO: Comment
'''

MODEL_PATH = "openai/gpt-oss-20b"
CACHE_DIR = "models"
REFERENCE_FILE = "RealSentences/BC5CDR.txt"

KG_TTL_PATH = "KG/BC5CDR_norm.ttl"

NUM_REF_SAMPLES = 25
NUM_LOOPS = 40

MAX_NEW_TOKENS = 100

OUTPUT_RAW_CSV = "Output/output_BC5CDR_121-3.csv"

TEMPERATURE_CHOICES = [0.01, 0.2, 0.4, 0.6, 0.8]

DOMAIN_WEIGHTS = {
    "AnatomicalStructure": 1/21,
    "Bacterium": 1/21,
    "BiologicFunction": 1/21,
    "BiomedicalOccupationOrDiscipline": 1/21,
    "BodySubstance": 1/21,
    "BodySystem": 1/21,
    "Chemical": 1/21,
    "ClinicalAttribute": 1/21,
    "Eukaryote": 1/21,
    "Finding": 1/21,
    "Food": 1/21,
    "HealthcareActivity": 1/21,
    "InjuryOrPoisoning": 1/21,
    "IntellectualProduct": 1/21,
    "MedicalDevice": 1/21,
    "Organization": 1/21,
    "PopulationGroup": 1/21,
    "ProfessionalOrOccupationalGroup": 1/21,
    "ResearchActivity": 1/21,
    "SpatialConcept": 1/21,
    "Virus": 1/21,
}

POWER_A = 121.0
POWER_B = -3.0
POWER_DEPTH = 30

PROMPT_TEMPLATE = """Generate synthetic data for a biomedical text classification dataset with twenty-one possible entity types.

ENTITY TYPES:
- AnatomicalStructure: Specific parts of the body or anatomical regions (e.g., left ventricle, femoral artery).
- Bacterium: Bacterial organisms mentioned in the text (e.g., Escherichia coli, Staphylococcus aureus).
- BiologicFunction: Biological or physiological processes and functions (e.g., immune response, hemostasis).
- BiomedicalOccupationOrDiscipline: Biomedical roles or disciplines (e.g., cardiologist, oncology, radiology).
- BodySubstance: Substances originating from the body (e.g., blood, plasma, cerebrospinal fluid).
- BodySystem: Functional body systems (e.g., cardiovascular system, respiratory system).
- Chemical: Chemical substances and compounds (e.g., ethanol, sodium chloride).
- ClinicalAttribute: Clinical characteristics or attributes (e.g., severity, stage II, BMI).
- Eukaryote: Eukaryotic organisms such as parasites or fungi (e.g., Plasmodium falciparum, Candida albicans).
- Finding: Clinical findings or observations (e.g., fever, rash, wheezing, elevated creatinine).
- Food: Food items or nutrients (e.g., milk, gluten, high-fat diet).
- HealthcareActivity: Healthcare-related activities not primarily procedures (e.g., nursing care, follow-up visit).
- InjuryOrPoisoning: Injuries, poisonings, and related conditions (e.g., blunt trauma, acetaminophen overdose).
- IntellectualProduct: Guidelines, questionnaires, reports, and other intellectual artifacts (e.g., clinical guideline, survey form).
- MedicalDevice: Medical instruments or devices (e.g., pacemaker, ventilator, stent).
- Organization: Institutions or organizations (e.g., hospital, research institute, WHO).
- PopulationGroup: Groups of people or patient populations (e.g., elderly patients, pediatric population).
- ProfessionalOrOccupationalGroup: Professional groups or categories (e.g., nurses, surgeons, laboratory technicians).
- ResearchActivity: Research-related activities or study types (e.g., randomized controlled trial, cohort study).
- SpatialConcept: Spatial or locational concepts relevant to medicine (e.g., upper quadrant, distal segment).
- Virus: Viral agents (e.g., influenza virus, SARS-CoV-2).

TASK:
- Write EXACTLY ONE English sentence.
- The sentence must be biomedical.
- No quotes, no lists, no extra formatting.
- Do not copy or paraphrase reference samples.
- Vary clinical domains and topics.
- At least one of the entity types above should be clearly instantiated in the sentence.

REFERENCE_SAMPLES:
{reference_block}
"""

class SkipNonWordTokens(LogitsProcessor):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.bad_token_ids = self._compute_bad_token_ids()

    def _compute_bad_token_ids(self):
        bad = []
        for tid in range(self.tokenizer.vocab_size):
            decoded = self.tokenizer.decode([tid], clean_up_tokenization_spaces=False)
            if decoded.strip() == "" or decoded in ["\n", "\r", "\t", "#"]:
                bad.append(tid)
        return set(bad)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        if self.bad_token_ids:
            scores[:, list(self.bad_token_ids)] = -float("inf")
        return scores


def compute_generated_logprob(model, full_ids: torch.LongTensor, prompt_len: int) -> float:
    device = next(model.parameters()).device
    full_ids = full_ids.to(device)
    attn = torch.ones_like(full_ids, device=device)
    with torch.no_grad():
        outputs = model(input_ids=full_ids, attention_mask=attn)
    logits = outputs.logits[:, :-1, :]
    target_ids = full_ids[:, 1:]
    logprobs = torch.log_softmax(logits, dim=-1)
    token_logprobs = logprobs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
    gen_start = prompt_len - 1
    return float(token_logprobs[:, gen_start:].sum().item())


def get_topk_candidate_token_ids(model, tokenizer, prefix_ids, num_candidates, skip_processor, top_k_pool=100):
    device = next(model.parameters()).device
    prefix_ids = prefix_ids.to(device)
    with torch.no_grad():
        outputs = model(input_ids=prefix_ids)
    logits = outputs.logits[:, -1, :][0]
    if skip_processor.bad_token_ids:
        logits[list(skip_processor.bad_token_ids)] = -float("inf")
    probs = F.softmax(logits, dim=-1)
    k = min(top_k_pool, probs.size(0))
    topk_probs, topk_ids = torch.topk(probs, k=k)
    candidate_ids = []
    seen = set()
    for _ in range(top_k_pool):
        if len(candidate_ids) >= num_candidates or topk_probs.sum() <= 0:
            break
        idx = torch.multinomial(topk_probs, num_samples=1).item()
        tid = topk_ids[idx].item()
        topk_probs[idx] = 0.0
        decoded = tokenizer.decode([tid], clean_up_tokenization_spaces=False)
        norm = decoded.strip().strip(".,;:!?\"'()").lower()
        if not norm or norm in seen:
            continue
        seen.add(norm)
        candidate_ids.append(tid)
    return candidate_ids


def branch_once(model, tokenizer, prefixes, branching_factor, skip_processor):
    device = next(model.parameters()).device
    new_prefixes = []
    for prefix_ids in prefixes:
        prefix_ids = prefix_ids.to(device)
        top_k_pool = max(branching_factor * 5, branching_factor + 100)
        candidate_token_ids = get_topk_candidate_token_ids(
            model, tokenizer, prefix_ids, branching_factor, skip_processor, top_k_pool=top_k_pool
        )
        for tid in candidate_token_ids:
            token_tensor = torch.tensor([[tid]], dtype=torch.long, device=device)
            branch_input_ids = torch.cat([prefix_ids, token_tensor], dim=1)
            new_prefixes.append(branch_input_ids)
    return new_prefixes


def complete_prefix(model, tokenizer, prefix_ids, max_new_tokens, skip_processor, prompt_len):
    device = next(model.parameters()).device
    prefix_ids = prefix_ids.to(device)
    attention = torch.ones_like(prefix_ids, device=device)
    temperature = random.choice(TEMPERATURE_CHOICES)
    logits_processors = LogitsProcessorList([skip_processor])
    with torch.no_grad():
        gen_output = model.generate(
            input_ids=prefix_ids,
            attention_mask=attention,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            num_beams=1,
            num_return_sequences=1,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            early_stopping=True,
            logits_processor=logits_processors,
        )
    full_ids = gen_output[0].unsqueeze(0)
    value = compute_generated_logprob(model, full_ids, prompt_len)
    return full_ids, value, temperature


def load_reference_sentences(path: str):
    p = Path(path)
    lines = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)
    return lines


def load_medical_entities(ttl_path: str):
    g = Graph()
    p = Path(ttl_path)
    if not p.exists():
        raise FileNotFoundError(f"TTL file not found: {ttl_path}")
    g.parse(str(p), format="turtle")

    parent = {}

    def find(x):
        if x not in parent:
            parent[x] = x
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a, b):
        pa = find(a)
        pb = find(b)
        if pa != pb:
            parent[pb] = pa

    for s, o in g.subject_objects(OWL.sameAs):
        union(s, o)

    canonical_info = {}

    for entity, _, domain in g.triples((None, RDF.type, None)):
        if (domain, RDF.type, RDFS.Class) not in g:
            continue
        label = g.value(entity, RDFS.label)
        if label is None:
            continue
        if entity in parent:
            canon = find(entity)
        else:
            canon = entity
        d = str(domain)
        if "#" in d:
            domain_name = d.split("#")[-1]
        else:
            domain_name = d.rsplit("/", 1)[-1]
        if canon not in canonical_info:
            canonical_info[canon] = (domain_name, str(label))

    entities_by_domain = defaultdict(list)
    for _, (domain_name, label) in canonical_info.items():
        entities_by_domain[domain_name].append(label)

    for d in list(entities_by_domain.keys()):
        entities_by_domain[d] = list(sorted(set(entities_by_domain[d])))

    return dict(entities_by_domain)


def sample_domain(domain_names, domain_weights):
    names = []
    weights = []
    for d in domain_names:
        if d in domain_weights:
            names.append(d)
            weights.append(domain_weights[d])
    if not names:
        idx = random.randint(0, len(domain_names) - 1)
        return domain_names[idx]
    s = sum(weights)
    weights = [w / s for w in weights]
    idx = np.random.choice(len(names), p=weights)
    return names[idx]


def build_branching_factors(a: float, b: float, depth_max: int):
    factors = []
    for i in range(1, depth_max):
        val = a * (i ** b)
        f = int(val)
        if f < 1:
            f = 1
        factors.append(f)
    while factors and factors[-1] == 1:
        factors.pop()
    return factors

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, cache_dir=CACHE_DIR, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        cache_dir=CACHE_DIR,
        device_map="auto",
        dtype="auto",
        trust_remote_code=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = next(model.parameters()).device

    reference_sentences = load_reference_sentences(REFERENCE_FILE)
    if len(reference_sentences) < NUM_REF_SAMPLES:
        raise ValueError(f"Not enough sentences in {REFERENCE_FILE}: need at least {NUM_REF_SAMPLES}")

    entities_by_domain = load_medical_entities(KG_TTL_PATH)
    if not entities_by_domain:
        raise ValueError(f"No entities found in TTL file {KG_TTL_PATH}")

    skip_processor = SkipNonWordTokens(tokenizer)

    branching_factors = build_branching_factors(POWER_A, POWER_B, POWER_DEPTH)
    if not branching_factors:
        raise ValueError("Branching factors list is empty; check POWER_A, POWER_B, POWER_DEPTH.")
    if len(branching_factors) > 1:
        branching_factors = branching_factors[1:]
    else:
        branching_factors = []

    domains_for_weights = list(DOMAIN_WEIGHTS.keys())
    if not domains_for_weights:
        raise ValueError("DOMAIN_WEIGHTS is empty.")

    num_roots = max(1, int(POWER_A))

    max_sentences_theoretical = num_roots
    for f in branching_factors:
        max_sentences_theoretical *= f
    MAX_SENTENCES = max_sentences_theoretical

    sentences_written = 0

    with open(OUTPUT_RAW_CSV, mode="w", newline="", encoding="utf-8") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["sentence", "value", "temperature", "domain"])

        for loop_idx in range(NUM_LOOPS):
            if sentences_written >= MAX_SENTENCES:
                break

            sampled_refs = random.sample(reference_sentences, NUM_REF_SAMPLES)
            reference_block = "\n".join(sampled_refs)
            base_prompt = PROMPT_TEMPLATE.format(reference_block=reference_block)

            weight_sum = sum(DOMAIN_WEIGHTS[d] for d in domains_for_weights)

            targets = {
                d: num_roots * (DOMAIN_WEIGHTS[d] / weight_sum)
                for d in domains_for_weights
            }

            allocations = {d: int(targets[d]) for d in domains_for_weights}

            assigned = sum(allocations.values())
            remaining = num_roots - assigned

            if remaining > 0:
                remainders = sorted(
                    domains_for_weights,
                    key=lambda d: (targets[d] - allocations[d]),
                    reverse=True,
                )
                for d in remainders[:remaining]:
                    allocations[d] += 1

            prefixes = []

            for domain, count in allocations.items():
                if count <= 0:
                    continue

                domain_entities = entities_by_domain.get(domain, [])
                for _ in range(count):
                    if domain_entities:
                        sampled_dom_entities = random.sample(domain_entities, min(10, len(domain_entities)))
                        domain_hint_block = (
                            "\nDOMAIN FOCUS:\n"
                            f"- This sentence should be clearly related to the domain: {domain}.\n\n"
                            "DOMAIN ENTITY SEEDS (do NOT copy them verbatim; you may use related clinical concepts "
                            "and optionally mention one or two of them):\n"
                            + "\n".join(f"- {e}" for e in sampled_dom_entities)
                            + "\n"
                        )
                    else:
                        domain_hint_block = (
                            "\nDOMAIN FOCUS:\n"
                            f"- This sentence should be clearly related to the domain: {domain}.\n\n"
                        )

                    specific_prompt = base_prompt + domain_hint_block

                    harmony_prompt = (
                        f"<|start|>user<|message|>{specific_prompt}<|end|>"
                        f"<|start|>assistant<|channel|>final<|message|>"
                    )

                    inputs = tokenizer(harmony_prompt, return_tensors="pt").to(device)
                    input_ids = inputs["input_ids"]
                    prompt_len = input_ids.size(1)

                    prefixes.append((input_ids, prompt_len, domain))

            if not prefixes:
                continue

            for depth, bf in enumerate(branching_factors):
                new_prefixes = []
                for prefix_ids, prompt_len, domain in prefixes:
                    prefix_ids = prefix_ids.to(device)
                    top_k_pool = max(bf * 5, bf + 100)
                    candidate_token_ids = get_topk_candidate_token_ids(
                        model, tokenizer, prefix_ids, bf, skip_processor, top_k_pool=top_k_pool
                    )
                    for tid in candidate_token_ids:
                        token_tensor = torch.tensor([[tid]], dtype=torch.long, device=device)
                        branch_input_ids = torch.cat([prefix_ids, token_tensor], dim=1)
                        new_prefixes.append((branch_input_ids, prompt_len, domain))
                prefixes = new_prefixes
                if not prefixes:
                    break

            for prefix_ids, prompt_len, domain in prefixes:
                if sentences_written >= MAX_SENTENCES:
                    break

                full_ids, value, temp = complete_prefix(
                    model, tokenizer, prefix_ids, MAX_NEW_TOKENS, skip_processor, prompt_len
                )

                gen_ids = full_ids[0, prompt_len:]
                text = tokenizer.decode(
                    gen_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                ).strip()
                if not text:
                    continue

                writer.writerow([text, value, temp, domain if domain is not None else ""])
                sentences_written += 1


if __name__ == "__main__":
    main()

