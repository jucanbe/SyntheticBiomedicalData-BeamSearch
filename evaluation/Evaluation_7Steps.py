import argparse, math, os, re, json, torch
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from scipy.stats import entropy, wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.feature_extraction.text import CountVectorizer
from transformers import AutoTokenizer, AutoModelForTokenClassification
from scipy.spatial import ConvexHull

'''
This script evaluates the synthetic dataset according to the Scoreboard paper.
TODO: finish comment the functions
'''

def load_real_sentences(path):
    '''
    Method to load the real sentences files
    :param path:
    :return:
    '''
    lines = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)
    return lines


def load_synthetic_csv(path):
    '''
    Method to load the synthetic data files
    :param path:
    :return:
    '''
    df = pd.read_csv(path)
    if "sentence" not in df.columns:
        raise ValueError("CSV must contain a 'sentence' column")
    df["sentence"] = df["sentence"].astype(str)
    return df


def load_entities(entities_dir):
    '''
    Method to load the entities files
    :param entities_dir:
    :return:
    '''
    ents = {}
    for fname in os.listdir(entities_dir):
        if not fname.lower().endswith(".txt"):
            continue
        domain = fname[:-4]
        path = os.path.join(entities_dir, fname)
        items = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(line)
        ents[domain] = items
    return ents


def build_embeddings_bert(model_dir, sentences, batch_size=32, max_length=256, device=None):
    '''
    Method to build the embeddings model
    :param model_dir:
    :param sentences:
    :param batch_size:
    :param max_length:
    :param device:
    :return:
    '''
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model_tc = AutoModelForTokenClassification.from_pretrained(model_dir)
    base_model = model_tc.bert
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model.to(device)
    base_model.eval()
    all_embs = []
    with torch.no_grad():
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i: i + batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            out = base_model(**enc)
            last = out.last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1)
            masked = last * mask
            emb = masked.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            all_embs.append(emb.cpu().numpy())
    if not all_embs:
        return np.zeros((0, base_model.config.hidden_size), dtype=np.float32)
    return np.concatenate(all_embs, axis=0)


def normalize_text(s):
    '''
    Method to normalize text
    :param s:
    :return:
    '''
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    return " " + s.strip() + " "


def detect_entities_lex(sentence, entities_by_domain):
    '''
    Method to detect entities in sentence
    :param sentence:
    :param entities_by_domain:
    :return:
    '''
    detected = defaultdict(list)
    s = normalize_text(sentence)
    for domain, ents in entities_by_domain.items():
        for e in ents:
            en = " " + e.lower().strip() + " "
            if en in s:
                detected[domain].append(e)
    return detected


def js_divergence(p, q, eps=1e-8):
    '''
    Method to calculate Jensen-Shannon divergence
    :param p:
    :param q:
    :param eps:
    :return:
    '''
    p = np.asarray(p, dtype=float) + eps
    q = np.asarray(q, dtype=float) + eps
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m) + entropy(q, m))


def hull_area_2d(points):
    '''
    Method to calculate area of polygon
    :param points:
    :return:
    '''
    if len(points) < 3:
        return float("nan")
    try:
        hull = ConvexHull(points)
        return float(hull.volume)
    except Exception:
        return float("nan")


def ngram_jaccard(a, b, n=4):
    '''
    Method to calculate Jaccard similarity
    :param a:
    :param b:
    :param n:
    :return:
    '''
    def ngrams(text):
        toks = re.findall(r"\w+", text.lower())
        return set(tuple(toks[i: i + n]) for i in range(max(0, len(toks) - n + 1)))
    A = ngrams(a)
    B = ngrams(b)
    if not A and not B:
        return 0.0
    return len(A & B) / max(1, len(A | B))

def compute_congruence(emb_real, emb_syn, n_samples=1000, n_bins=50):
    '''
    Method to compute congruence criteria
    :param emb_real:
    :param emb_syn:
    :param n_samples:
    :param n_bins:
    :return:
    '''
    if len(emb_real) == 0 or len(emb_syn) == 0:
        return {
            "cosine_similarity_embedding_mean": float("nan"),
            "js_divergence_pairwise_distances": float("nan"),
            "earth_movers_distance_wasserstein": float("nan"),
        }
    n_r = min(len(emb_real), n_samples)
    n_s = min(len(emb_syn), n_samples)
    idx_r = np.random.choice(len(emb_real), n_r, replace=False)
    idx_s = np.random.choice(len(emb_syn), n_s, replace=False)
    r = emb_real[idx_r]
    s = emb_syn[idx_s]
    mean_r = r.mean(axis=0)
    mean_s = s.mean(axis=0)
    cos_sim_means = 1 - cosine(mean_r, mean_s)

    dists_real = pairwise_distances(r, r, metric="cosine")
    dists_syn = pairwise_distances(s, s, metric="cosine")
    iu_r = np.triu_indices_from(dists_real, k=1)
    iu_s = np.triu_indices_from(dists_syn, k=1)
    dr_flat = dists_real[iu_r]
    ds_flat = dists_syn[iu_s]

    hist_r, bins = np.histogram(dr_flat, bins=n_bins, range=(0, 2), density=True)
    hist_s, _ = np.histogram(ds_flat, bins=bins, density=True)
    jsd = js_divergence(hist_r, hist_s)

    try:
        wdist = float(wasserstein_distance(dr_flat, ds_flat))
    except Exception:
        wdist = float("nan")

    return {
        "cosine_similarity_embedding_mean": float(cos_sim_means),
        "js_divergence_pairwise_distances": float(jsd),
        "earth_movers_distance_wasserstein": wdist,
    }

def compute_coverage(real_sentences, syn_sentences, emb_real, emb_syn,
                     n_real_sample=5000, nn_threshold=0.8):
    '''
    Method to compute coverage criteria
    :param real_sentences:
    :param syn_sentences:
    :param emb_real:
    :param emb_syn:
    :param n_real_sample:
    :param nn_threshold:
    :return:
    '''

    vect_real = CountVectorizer(lowercase=True, token_pattern=r"[A-Za-z][A-Za-z0-9_\-]+", min_df=5)
    vect_real.fit(real_sentences)
    real_vocab = set(vect_real.get_feature_names_out())

    vect_syn = CountVectorizer(lowercase=True, token_pattern=r"[A-Za-z][A-Za-z0-9_\-]+")
    vect_syn.fit(syn_sentences)
    syn_vocab = set(vect_syn.get_feature_names_out())

    vocab_recall = (len(real_vocab & syn_vocab) / max(1, len(real_vocab))) * 100.0   # %

    if len(emb_real) > 2 and len(emb_syn) > 2:
        pca = PCA(n_components=2, random_state=0)
        r2 = pca.fit_transform(emb_real)
        s2 = pca.transform(emb_syn)
        area_real = hull_area_2d(r2)
        area_syn = hull_area_2d(s2)
        hull_ratio = area_syn / area_real if area_real and not math.isnan(area_real) else float("nan")
    else:
        hull_ratio = float("nan")

    if len(emb_real) == 0 or len(emb_syn) == 0:
        nn_coverage_pct = float("nan")
    else:
        n_r = min(len(emb_real), n_real_sample)
        idx_r = np.random.choice(len(emb_real), n_r, replace=False)
        r = emb_real[idx_r]
        dists = pairwise_distances(r, emb_syn, metric="cosine")
        nearest_sim = 1 - dists.min(axis=1)
        nn_coverage_pct = float((nearest_sim >= nn_threshold).mean() * 100.0)  # %

    return {
        "vocab_recall_pct": float(vocab_recall),
        "convex_hull_ratio_syn_over_real_pca": float(hull_ratio),
        "coverage_nn_similarity_ge_0.8_pct": float(nn_coverage_pct),
    }

def compute_constraints(df_syn):
    '''
    Method to compute constraints criteria
    :param df_syn:
    :return:
    '''
    s = df_syn["sentence"].astype(str)
    non_empty = s.str.strip().str.len() > 0
    has_period = s.str.contains(r"\.")
    single_sentence = s.str.count(r"\.") <= 1

    valid = non_empty & has_period & single_sentence

    return {
        "constraint_violation_rate_pct": float((1.0 - valid.mean()) * 100.0),
        "constraint_valid_samples_pct": float(valid.mean() * 100.0),
    }


def compute_completeness(df_syn, entities_by_domain):
    '''
    Method to compute completeness criteria
    :param df_syn:
    :param entities_by_domain:
    :return:
    '''
    sents = df_syn["sentence"].astype(str).tolist()
    ent_counts = []

    for s in sents:
        det = detect_entities_lex(s, entities_by_domain)
        ent_counts.append(sum(len(v) for v in det.values()))

    ent_counts = np.array(ent_counts, dtype=float)

    has_required = ent_counts > 0

    proportion_required_pct = has_required.mean() * 100.0
    missing_pct = (1.0 - has_required.mean()) * 100.0

    return {
        "proportion_required_clinical_field_pct": float(proportion_required_pct),
        "missing_data_percentage_required_field_pct": float(missing_pct),
    }


def compute_compliance(real_sentences, syn_sentences, emb_real, emb_syn,
                       n_real_sample=5000, n_syn_sample=1000,
                       high_sim_threshold=0.95, jaccard_threshold=0.8):
    '''
    Method to compute compliance criteria
    :param real_sentences:
    :param syn_sentences:
    :param emb_real:
    :param emb_syn:
    :param n_real_sample:
    :param n_syn_sample:
    :param high_sim_threshold:
    :param jaccard_threshold:
    :return:
    '''

    if len(real_sentences) == 0 or len(syn_sentences) == 0:
        return {
            "nearest_real_cosine_similarity_mean": float("nan"),
            "suspected_memorised_samples_pct": float("nan"),
        }

    n_r = min(len(real_sentences), n_real_sample)
    idx_r = np.random.choice(len(real_sentences), n_r, replace=False)
    r_sents = [real_sentences[i] for i in idx_r]
    r_emb = emb_real[idx_r]

    n_s = min(len(syn_sentences), n_syn_sample)
    syn_idx = np.random.choice(len(syn_sentences), n_s, replace=False)
    s_sample = [syn_sentences[i] for i in syn_idx]
    s_emb = emb_syn[syn_idx]

    dists = pairwise_distances(s_emb, r_emb, metric="cosine")
    nearest_idx = dists.argmin(axis=1)
    nearest_sim = 1 - dists[np.arange(len(s_sample)), nearest_idx]

    high_sim = nearest_sim > high_sim_threshold

    jaccs = []
    for i, flag in enumerate(high_sim):
        if not flag:
            jaccs.append(0.0)
            continue
        s = s_sample[i]
        r = r_sents[nearest_idx[i]]
        jaccs.append(ngram_jaccard(s, r, n=4))

    jaccs = np.array(jaccs)
    suspected_pct = float((jaccs > jaccard_threshold).mean() * 100.0)

    return {
        "nearest_real_cosine_similarity_mean": float(nearest_sim.mean()),
        "suspected_memorised_samples_pct": suspected_pct,
    }


def compute_consistency_inferred(syn_sentences, emb_syn, entities_by_domain):
    '''
    Method to compute consistency criteria
    :param syn_sentences:
    :param emb_syn:
    :param entities_by_domain:
    :return:
    '''
    if len(emb_syn) == 0 or not entities_by_domain:
        return {
            "variance_domain_cosine_sim_to_global": float("nan"),
            "maxmin_domain_cosine_sim_to_global": float("nan"),
            "per_domain": {}
        }

    domain_to_indices = defaultdict(list)
    for i, s in enumerate(syn_sentences):
        det = detect_entities_lex(s, entities_by_domain)
        for d in det.keys():
            domain_to_indices[d].append(i)

    results_per_domain = {}
    cos_values = []
    global_mean = emb_syn.mean(axis=0)

    for d, idx_list in domain_to_indices.items():
        if len(idx_list) < 5:
            continue
        idx = np.array(idx_list, dtype=int)
        emb_d = emb_syn[idx]
        mean_d = emb_d.mean(axis=0)
        cos_sim = 1 - cosine(mean_d, global_mean)
        cos_values.append(cos_sim)
        results_per_domain[d] = {
            "cosine_sim_to_global_mean": float(cos_sim),
            "num_sentences": len(idx)
        }

    if cos_values:
        var_cos = float(np.var(cos_values))
        maxmin_cos = float(max(cos_values) - min(cos_values))
    else:
        var_cos = float("nan")
        maxmin_cos = float("nan")

    return {
        "variance_domain_cosine_sim_to_global": var_cos,
        "maxmin_domain_cosine_sim_to_global": maxmin_cos,
        "per_domain": results_per_domain
    }

def round_floats(obj, ndigits=4):
    '''
    Method to round floats
    :param obj:
    :param ndigits:
    :return:
    '''
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: round_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_floats(v, ndigits) for v in obj]
    return obj

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic_csv", required=True)
    parser.add_argument("--real_txt", required=True)
    parser.add_argument("--bert_dir", required=True)
    parser.add_argument("--entities_dir", required=True)
    args = parser.parse_args()

    real_sentences = load_real_sentences(args.real_txt)
    df_syn = load_synthetic_csv(args.synthetic_csv)
    syn_sentences = df_syn["sentence"].astype(str).tolist()
    entities_by_domain = load_entities(args.entities_dir)

    emb_real = build_embeddings_bert(args.bert_dir, real_sentences)
    emb_syn = build_embeddings_bert(args.bert_dir, syn_sentences)

    cong = round_floats(compute_congruence(emb_real, emb_syn), 4)
    cov = round_floats(compute_coverage(real_sentences, syn_sentences, emb_real, emb_syn), 4)
    constraints = round_floats(compute_constraints(df_syn), 4)
    compltn = round_floats(compute_completeness(df_syn, entities_by_domain), 4)
    compl = round_floats(compute_compliance(real_sentences, syn_sentences, emb_real, emb_syn), 4)
    consis = compute_consistency_inferred(syn_sentences, emb_syn, entities_by_domain)

    print("=== 1C - CONGRUENCE")
    print(json.dumps(cong, indent=2))

    print("\n=== 2C - COVERAGE ===")
    print(json.dumps(cov, indent=2))

    print("\n=== 3C - CONSTRAINT ===")
    print(json.dumps(constraints, indent=2))

    print("\n=== 4C - COMPLETENESS ===")
    print(json.dumps(compltn, indent=2))

    print("\n=== 5C - COMPLIANCE ===")
    print(json.dumps(compl, indent=2))

    print("\n=== 7C - CONSISTENCY ===")
    print(json.dumps(consis, indent=2))


if __name__ == "__main__":
    main()
