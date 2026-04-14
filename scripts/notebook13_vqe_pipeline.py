# =============================================================================
# QBIO PIPELINE — NOTEBOOK 13 VQE INTEGRATED
# Full NVIDIA hosted biology pipeline + Marena 2026 6-31G* H-bond VQE scorer
# Tommaso R. Marena | The Catholic University of America | April 2026
#
# HOW TO RUN:
#   1. Add NVIDIA_API_KEY to Colab Secrets (left sidebar -> key icon)
#   2. Enable notebook access for the secret
#   3. Runtime -> Change runtime type -> CPU -> High-RAM -> Save
#   4. Paste this entire script into ONE Colab cell
#   5. Runtime -> Run cell
#   6. First run: ~8-12 hours (VQE dominates)
#      Subsequent runs: cache in /content/qbio_pipeline/cache/ skips VQE
#
# OUTPUTS:
#   /content/qbio_pipeline/outputs/quantum_scores.csv
#   /content/qbio_pipeline/outputs/pipeline_report.json
#   /content/qbio_pipeline/outputs/pipeline_report.csv
#   /content/qbio_pipeline/outputs/*_hbond_candidates.csv
#
# CHANGES FROM ORIGINAL (NB13 final):
#   - Cache NOT cleared on startup (preserves monomer/dimer results)
#   - looks_like_pdb() requires string to START with ATOM/HETATM/MODEL
#   - Angular filter: da > 100 AND aa > 100 before H-bond candidate append
#   - Drive auto-backup after pipeline completes
#   - max_memory=40000 (40 GB) for PySCF
# =============================================================================

import sys, subprocess, importlib, time, itertools, warnings, os, json, re, asyncio
import zipfile, shutil
from pathlib import Path
warnings.filterwarnings('ignore')

# Cache: preserve existing results, only create dir if missing
_CACHE_PATH = Path('/content/qbio_pipeline/cache')
if not _CACHE_PATH.exists():
    _CACHE_PATH.mkdir(parents=True, exist_ok=True)
    print(f'[CACHE DIR CREATED] {_CACHE_PATH}')
else:
    print(f'[CACHE EXISTS] {_CACHE_PATH} — reusing')


def ensure(import_name, pip_name=None):
    pip_name = pip_name or import_name
    try:
        importlib.import_module(import_name)
    except ImportError:
        print(f'[INSTALL] {pip_name}...', flush=True)
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', pip_name])
        print(f'[DONE] {pip_name}', flush=True)


t_global = time.time()
for pkg in [
    'numpy', 'matplotlib', 'requests', 'httpx', 'pandas',
    ('nest_asyncio', 'nest_asyncio'),
    ('scipy',            'scipy'),
    ('pyscf',            'pyscf'),
    ('openfermion',      'openfermion'),
    ('openfermionpyscf', 'openfermionpyscf'),
    ('qiskit',           'qiskit'),
    ('qiskit_algorithms','qiskit-algorithms'),
]:
    ensure(*pkg) if isinstance(pkg, tuple) else ensure(pkg)

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import requests, httpx, pandas as pd
import nest_asyncio
from scipy.stats import spearmanr
from scipy.sparse.linalg import eigsh as sparse_eigsh
from pyscf import gto, scf, mcscf, ao2mo
from pyscf.fci import direct_spin1, cistring
from openfermion.ops import InteractionOperator
from openfermion.transforms import jordan_wigner
from openfermion.linalg import get_sparse_operator
from openfermion import get_fermion_operator
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit.circuit.library import EfficientSU2
from qiskit_algorithms.minimum_eigensolvers import VQE
from qiskit_algorithms.optimizers import SLSQP
nest_asyncio.apply()
print(f'[OK] All imports | {time.time()-t_global:.1f}s')


def load_nvidia_api_key():
    try:
        from google.colab import userdata
    except ImportError:
        raise RuntimeError("Run this in Google Colab.")
    key = userdata.get("NVIDIA_API_KEY")
    if not key or not key.startswith("nvapi-"):
        raise RuntimeError("Invalid NVIDIA_API_KEY in Colab Secrets.")
    os.environ["NVIDIA_API_KEY"] = key
    return key


NVIDIA_API_KEY = load_nvidia_api_key()
print("Loaded NVIDIA_API_KEY:", NVIDIA_API_KEY[:10] + "...")

CONFIG = {
    "base_dir": "/content/qbio_pipeline",
    "run": {
        "evo2": True, "esm2": True, "rfdiffusion": True,
        "openfold3": True, "boltz2": True, "diffdock": True,
        "molmim": True, "genmol": True, "quantum_rescore": True,
    },
    "protein_sequence": (
        "MDILCEENTSLSSTTNSLMQLNDDTRLYSNDFNSGEANTSDAFNWTVDSENRTNLSCEGCLSPSCLSLLHLQEKNW"
        "SALLTAVVIILTIAGNILVIMAVSLEKKLQNATNYFLMSLAIADMLLGFLVMPVSMLTILYGYRWPLPSKLCAVWIY"
        "LDVLFSTASIMHLCAISLDRYVAIQNPIHHSRFNSRTKAFLKIIAVWTISVGISMPIPVFGLQDDSKVFKEGSCLLA"
        "DDNFVLIGSFVSFFIPLTIMVITYFLTIKSLQKEATLCVSDLGTRAKLASFSFLPQSSLSSEKLFQRSIHREPGSYT"
        "GRRTMQSISNEQKACKVLGIVFFLFVVMWCPFFITNIMAVICKESCNEDVIGALLNVFVWIGYLSSAVNPLVYTLFN"
        "KTYRSAFSRYIQCQYKENKKPLQLILVNTIPALAYKSSQLQMGQKKNSKQDAKTTDNDCSMVALGKQHSEEASKDNS"
        "DGVNEKVSCV"
    ),
    "dna_seq_1": "AGGAACACGTGACCC",
    "dna_seq_2": "TGGGTCACGTGTTCC",
    "dna_seed_sequence": "ACTGACTGACTGACTG",
    "evo2_num_tokens": 8, "evo2_top_k": 1,
    "protein_pdb_id": "8G43",
    "ligand_rcsb_code": "ZU6",
    "ligand_smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "rfdiffusion_pdb_id": "1R42",
    "rfdiffusion_contigs": "A20-60/0 50-100",
    "rfdiffusion_hotspots": ["A50", "A51", "A52", "A53", "A54"],
    "rfdiffusion_steps": 15,
    "diffdock_num_poses": 10, "diffdock_time_divisions": 20, "diffdock_steps": 18,
    "molmim_seed_smiles": "[H][C@@]12Cc3c[nH]c4cccc(C1=C[C@H](NC(=O)N(CC)CC)CN2C)c34",
    "molmim_property_name": "QED", "molmim_num_molecules": 30,
    "genmol_smiles": "C124CN3C1.S3(=O)(=O)CC.C4C#N.[*{20-20}]",
    "genmol_num_molecules": 30, "genmol_scoring": "QED",
    "hb_distance_cutoff": 3.5,
    "hb_top_k": 10,
    "quantum_top_k_contacts": 5,
    "vqe_basis": "6-31g*",
    "vqe_ncas": 8,
    "vqe_nelecas": 8,
    "vqe_reps": 2,
    "vqe_seeds": 3,
    "vqe_maxiter": 600,
    "vqe_cache": True,
}

BASE  = Path(CONFIG["base_dir"])
RAW   = BASE / "raw"
OUT   = BASE / "outputs"
CACHE = BASE / "cache"
for p in [BASE, RAW, OUT, CACHE]:
    p.mkdir(parents=True, exist_ok=True)

AUTH = {"Authorization": f"Bearer {NVIDIA_API_KEY}"}


def now(): return time.strftime("%Y-%m-%d %H:%M:%S")
def save_text(path, text): Path(path).write_text(text); return str(path)
def save_bytes(path, content): Path(path).write_bytes(content); return str(path)
def safe_json_loads(text):
    try: return json.loads(text)
    except Exception: return None
def download_text(url, timeout=60):
    r = requests.get(url, timeout=timeout); r.raise_for_status(); return r.text
def rcsb_pdb(pdb_id):
    txt = download_text(f"https://files.rcsb.org/download/{pdb_id}.pdb")
    return "\n".join(l for l in txt.splitlines() if l.startswith(("ATOM","HETATM")))
def rcsb_ligand_sdf(code):
    return download_text(f"https://files.rcsb.org/ligands/download/{code}_ideal.sdf")
def extract_zip_if_needed(path):
    path = Path(path)
    if path.suffix != ".zip": return None
    target = path.with_suffix(""); target.mkdir(exist_ok=True)
    with zipfile.ZipFile(path,"r") as zf: zf.extractall(target)
    return str(target)
def save_response_content(resp, stem):
    ctype = resp.headers.get("Content-Type","")
    txt = ""
    try: txt = resp.text
    except Exception: pass
    if "application/json" in ctype or txt.strip().startswith(("{","[")):
        out = RAW / f"{stem}.json"; save_text(out, txt)
        return {"path": str(out), "content_type": ctype}
    elif "application/zip" in ctype:
        out = RAW / f"{stem}.zip"; save_bytes(out, resp.content)
        extracted = extract_zip_if_needed(out)
        return {"path": str(out), "extracted": extracted, "content_type": ctype}
    else:
        out = RAW / f"{stem}.bin"; save_bytes(out, resp.content)
        return {"path": str(out), "content_type": ctype}
def post_json(url, payload, extra_headers=None, timeout=1800):
    headers = {"Content-Type": "application/json", **AUTH}
    if extra_headers: headers.update(extra_headers)
    return requests.post(url, headers=headers, json=payload, timeout=timeout)
async def nvcf_post_json(url, payload, poll_seconds=300, timeout=2400):
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json",
        "NVCF-POLL-SECONDS": str(poll_seconds),
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, headers=headers, json=payload)
        if resp.status_code == 200: return resp
        if resp.status_code == 202:
            task_id = resp.headers.get("nvcf-reqid")
            if not task_id: raise RuntimeError(f"202 no nvcf-reqid: {resp.text}")
            status_url = f"https://api.nvcf.nvidia.com/v2/nvcf/pexec/status/{task_id}"
            while True:
                s = await client.get(status_url, headers=headers)
                if s.status_code == 200: return s
                if s.status_code in (202,204): await asyncio.sleep(10); continue
                raise RuntimeError(f"NVCF poll failed: {s.status_code} {s.text}")
        raise RuntimeError(f"Request failed: {resp.status_code} {resp.text}")


# FIX: looks_like_pdb must START with ATOM/HETATM/MODEL, not merely contain it
def looks_like_pdb(x):
    if not isinstance(x, str): return False
    stripped = x.strip()
    return (stripped.startswith("ATOM  ") or
            stripped.startswith("HETATM") or
            stripped.startswith("MODEL"))


def recursive_find_pdb(obj, path="root", found=None):
    if found is None: found = []
    if isinstance(obj, dict):
        for k,v in obj.items(): recursive_find_pdb(v, f"{path}.{k}", found)
    elif isinstance(obj, list):
        for i,v in enumerate(obj): recursive_find_pdb(v, f"{path}[{i}]", found)
    elif looks_like_pdb(obj): found.append((path, obj))
    return found


def extract_pdb_from_json(json_path, prefix):
    path = Path(json_path)
    if not path.exists(): return []
    text = path.read_text()
    if looks_like_pdb(text):
        out = RAW / f"{prefix}_direct_0.pdb"; out.write_text(text)
        return [str(out)]
    data = safe_json_loads(text)
    if data is None: return []
    hits = recursive_find_pdb(data)
    out_paths = []
    for idx, (_, pdb_str) in enumerate(hits):
        out = RAW / f"{prefix}_extracted_{idx}.pdb"; out.write_text(pdb_str)
        out_paths.append(str(out))
    return out_paths


def parse_pdb_atoms(pdb_text):
    atoms = []
    for line in pdb_text.splitlines():
        if line.startswith(("ATOM","HETATM")):
            try:
                atoms.append({
                    "atom": line[12:16].strip(), "resn": line[17:20].strip(),
                    "chain": line[21].strip(), "resi": line[22:26].strip(),
                    "x": float(line[30:38]), "y": float(line[38:46]),
                    "z": float(line[46:54]),
                })
            except Exception: pass
    return atoms


def vec(a,b): return np.array([b["x"]-a["x"], b["y"]-a["y"], b["z"]-a["z"]],dtype=float)
def dist(a,b): return float(np.linalg.norm(vec(a,b)))
def angle_deg(v1,v2):
    n1,n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1<1e-8 or n2<1e-8: return None
    return float(np.degrees(np.arccos(np.clip(np.dot(v1,v2)/(n1*n2),-1,1))))


def build_backbone_index(atoms):
    res = {}
    for a in atoms:
        key=(a["chain"],a["resi"],a["resn"]); res.setdefault(key,[]).append(a)
    index = []
    for (chain,resi,resn),ra in res.items():
        def abn(name): return next((a for a in ra if a["atom"]==name),None)
        index.append({"chain":chain,"resi":resi,"resn":resn,
                      "N":abn("N"),"CA":abn("CA"),"C":abn("C"),"O":abn("O")})
    def rkey(x):
        try: return int(re.sub(r"[^0-9-]","",x["resi"]))
        except: return 0
    index.sort(key=lambda x:(x["chain"],rkey(x)))
    return index


def resi_sep(a,b):
    if a["chain"]!=b["chain"]: return 999
    try: return abs(int(re.sub(r"[^0-9-]","",a["resi"]))-int(re.sub(r"[^0-9-]","",b["resi"])))
    except: return 999


def infer_hbond_candidates(pdb_text, cutoff=3.5):
    index = build_backbone_index(parse_pdb_atoms(pdb_text))
    cands = []
    for d in index:
        if d["N"] is None or d["CA"] is None: continue
        for a in index:
            if a["O"] is None or a["C"] is None: continue
            if d["chain"]==a["chain"] and d["resi"]==a["resi"]: continue
            if resi_sep(d,a) <= 2: continue
            d_no = dist(d["N"], a["O"])
            if d_no > cutoff: continue
            da = angle_deg(vec(d["CA"],d["N"]), vec(d["N"],a["O"]))
            aa = angle_deg(vec(a["C"],a["O"]),  vec(a["O"],d["N"]))
            # Angular filter: require geometrically valid H-bond angles
            if da is None or da < 100.0: continue
            if aa is None or aa < 100.0: continue
            score = max(0.0,cutoff-d_no)*2.0
            if da: score += max(0.0,da-90.0)/30.0
            if aa: score += max(0.0,aa-90.0)/30.0
            cands.append({
                "donor_chain":d["chain"],"donor_resi":d["resi"],"donor_resn":d["resn"],
                "acceptor_chain":a["chain"],"acceptor_resi":a["resi"],"acceptor_resn":a["resn"],
                "distance_NO":round(d_no,4),
                "donor_angle_deg":None if da is None else round(da,2),
                "acceptor_angle_deg":None if aa is None else round(aa,2),
                "heuristic_score":round(score,4),
            })
    cands.sort(key=lambda x:(-x["heuristic_score"],x["distance_NO"]))
    return cands


# [... remainder of script unchanged from paste.txt — VQE core, pipeline orchestrator, main ...]
# Full script available in conversation history and Colab notebook
