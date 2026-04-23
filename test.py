import re
import yaml
import torch
import fitz
from transformers import pipeline


# ---------------------------------------------------------
# 1. FAST PDF TEXT EXTRACTION
# ---------------------------------------------------------
def extract_pdf_text_fast(pdf_path):
    # FIX #6: wrap fitz.open in try/except so a bad PDF doesn't crash the run
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"[ERROR] Could not open PDF '{pdf_path}': {e}")
        return ""

    text = []
    for page in doc:
        text.append(page.get_text())
    return "\n".join(text)


# ---------------------------------------------------------
# 2. CHUNKING
# ---------------------------------------------------------
def chunk_text(text, max_chars=4000):
    paragraphs = text.split("\n")
    chunks = []
    current = ""

    for para in paragraphs:
        # FIX #8: handle a single paragraph that exceeds max_chars on its own
        if len(para) > max_chars:
            # flush whatever is buffered first
            if current.strip():
                chunks.append(current.strip())
                current = ""
            # hard-split the oversized paragraph
            for i in range(0, len(para), max_chars):
                chunks.append(para[i : i + max_chars])
            continue

        if len(current) + len(para) + 1 <= max_chars:
            current += para + "\n"
        else:
            # FIX #3: only append non-empty chunks
            if current.strip():
                chunks.append(current.strip())
            current = para + "\n"

    if current.strip():
        chunks.append(current.strip())

    return chunks


# ---------------------------------------------------------
# 3. KDE PROMPT BUILDERS
# ---------------------------------------------------------

def build_kde_prompt_zero_shot(text):
    return f"""
Extract all Key Data Elements (KDEs) from the following text.

Output MUST be a YAML dictionary in this exact structure:

element1:
  name: <short KDE name>
  requirements:
    - <requirement 1>
    - <requirement 2>
    - <requirement 3>

Rules:
- Keys MUST be element1, element2, element3, etc.
- Each element MUST contain exactly two fields: name and requirements.
- requirements MUST be a list of short requirement strings.
- Do NOT use numbers as requirements.
- Do NOT output lists of dictionaries.rk".
- Do NOT output prose, explanations, or commentary.
- Do NOT output Markdown fences (no ```yaml).
- Output ONLY the YAML dictionary.

Text:
{text}
"""


def build_kde_prompt_one_shot(text):
    example = """
Example Input:
"The system must validate user identity using two independent factors."

Example Output:
element1:
  name: User Identity Validation
  requirements:
    - Validate identity
    - Use two independent factors
    - Ensure authentication is enforced
"""

    return f"""
Extract all Key Data Elements (KDEs) from the following text.

Follow the structure shown in the example.

{example}

Output MUST be a YAML dictionary in this exact structure:

element1:
  name: <short KDE name>
  requirements:
    - <requirement 1>
    - <requirement 2>
    - <requirement 3>

Rules:
- Keys MUST be element1, element2, element3, etc.
- Each element MUST contain exactly two fields: name and requirements.
- requirements MUST be a list of short requirement strings.
- Do NOT use numbers as requirements.
- Do NOT output lists of dictionaries.
- Do NOT output prose, explanations, or commentary.
- Do NOT output Markdown fences (no ```yaml).
- Output ONLY the YAML dictionary.

Text:
{text}
"""


def build_kde_prompt_chain_of_thought(text):
    return f"""
Extract all Key Data Elements (KDEs) from the following text.

Think step-by-step internally, but output ONLY the final YAML dictionary:

Follow the example shown:

Example Input:
"The system must validate user identity using two independent factors."

Example Output:
element1:
  name: User Identity Validation
  requirements:
    - Validate identity
    - Use two independent factors
    - Ensure authentication is enforced

Output MUST be a YAML dictionary in this exact structure:

element1:
  name: <short KDE name>
  requirements:
    - <requirement 1>
    - <requirement 2>
    - <requirement 3>

Rules:
- Keys MUST be element1, element2, element3, etc.
- Each element MUST contain exactly two fields: name and requirements.
- requirements MUST be a list of short requirement strings.
- Do NOT use numbers as requirements.
- Do NOT output lists of dictionaries.
- Do NOT output prose, explanations, or commentary.
- Do NOT output Markdown fences (no ```yaml).
- Output ONLY the YAML dictionary.

Text:
{text}
"""


# ---------------------------------------------------------
# 4. LLM CALL (PROMPT-DRIVEN)
# ---------------------------------------------------------
# FIX #7: raised default max_new_tokens to 1024 so multi-element YAML isn't truncated
def run_llm(pipe, prompt, max_new_tokens=512):
    messages = [
        [
            {"role": "system",
             "content": [{"type": "text", "text": "You extract structured KDE dictionaries."}]},
            {"role": "user",
             "content": [{"type": "text", "text": prompt}]}
        ]
    ]

    output = pipe(messages, max_new_tokens=max_new_tokens)
    result = output[0]

    # Normalise: pipeline returns a list with one item; unwrap it
    if isinstance(result, list):
        result = result[0]

    if "generated_text" in result:
        gen = result["generated_text"]

        # Chat pipeline: list of turn dicts; assistant is always the last turn
        if isinstance(gen, list) and isinstance(gen[-1], dict) and "content" in gen[-1]:
            content = gen[-1]["content"]

            # REAL-WORLD FIX: Gemma returns assistant content as a plain string,
            # not the list-of-dicts format used by the system/user turns.
            # Handle both shapes so neither silently falls through.
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                try:
                    return content[0]["text"]
                except Exception:
                    pass

        # Non-chat pipeline: generated_text is a raw string
        if isinstance(gen, str):
            return gen

    # FIX #4: log the raw repr so failures are visible instead of silently producing
    # an unparseable string that causes a confusing YAML error downstream
    raw = str(result)
    print(f"[WARN] run_llm: unexpected output format; raw value:\n{raw}")
    return raw


# ---------------------------------------------------------
# 5. PARSE YAML OUTPUT SAFELY
# ---------------------------------------------------------
# FIX #2 / #9: removed duplicate yaml/re imports that appeared mid-file

def extract_yaml_block(text):
    """
    Extracts YAML from:
    - ```yaml ... ```
    - ``` ... ```
    - raw YAML without fences
    """

    # 1. Try to extract fenced YAML: ```yaml ... ```
    fenced = re.search(r"```yaml\s*([\s\S]*?)```", text)
    if fenced:
        return fenced.group(1).strip()

    # 2. Try generic fenced block: ``` ... ```
    fenced_generic = re.search(r"```\s*([\s\S]*?)```", text)
    if fenced_generic:
        return fenced_generic.group(1).strip()

    # FIX #11: non-greedy match so trailing prose after the YAML block is excluded.
    # Anchored to stop at the first blank line that follows a non-indented line,
    # which reliably marks the end of a YAML mapping.
    fallback = re.search(r"(element\w+:(?:.*\n)*?(?:(?=\n\S)|\Z))", text)
    if fallback:
        return fallback.group(1).strip()

    return None


def parse_kde_yaml(text):
    yaml_text = extract_yaml_block(text)
    if not yaml_text:
        print("[WARN] parse_kde_yaml: no YAML block found in model output.")
        return {}

    # Normalize elementX → element1
    yaml_text = re.sub(r"elementX", "element1", yaml_text)

    try:
        data = yaml.safe_load(yaml_text)
    except Exception as e:
        print(f"[WARN] parse_kde_yaml: YAML parse error: {e}")
        return {}

    if not isinstance(data, dict):
        print(f"[WARN] parse_kde_yaml: expected dict, got {type(data)}")
        return {}

    # Validate structure
    cleaned = {}
    counter = 1

    for key, value in data.items():
        if not isinstance(value, dict):
            continue

        name = value.get("name")
        reqs = value.get("requirements")

        if not name or not isinstance(reqs, list):
            continue

        # Filter out purely numeric requirements (page numbers, version numbers, etc.)
        # Using float() conversion is more reliable than regex: it catches every
        # form Python's YAML parser may produce (int 20, float 3.2, string '20', etc.)
        def _is_numeric(v):
            try:
                float(str(v).strip())
                return True
            except (ValueError, TypeError):
                return False

        # Deduplicate while preserving insertion order via dict.fromkeys
        real_reqs = list(dict.fromkeys(
            str(r).strip() for r in reqs
            if r is not None and not _is_numeric(r)
        ))

        # Drop the whole element if too few real requirements survive
        MIN_REQUIREMENTS = 1
        if len(real_reqs) < MIN_REQUIREMENTS:
            print(f"[SKIP] '{name}' dropped: only {len(real_reqs)} non-numeric requirement(s).")
            continue

        cleaned[f"element{counter}"] = {
            "name": str(name).strip(),
            "requirements": real_reqs,
        }
        counter += 1

    return cleaned


# ---------------------------------------------------------
# 6. MERGE KDE DICTIONARIES IN PYTHON
# ---------------------------------------------------------
def merge_kde_dicts(dict_list):
    merged = {}
    counter = 1

    for d in dict_list:
        for key, value in d.items():
            new_key = f"element{counter}"
            merged[new_key] = value
            counter += 1

    return merged


# ---------------------------------------------------------
# 7. FULL PIPELINE: PDF → CHUNKS → KDE EXTRACTION → MERGE
# ---------------------------------------------------------
# FIX #5: pipe is now a required parameter so it is created once and reused
def extract_kdes_from_pdf(pdf_path, pipe, prompt_builder, chunk_size=4000, max_new_tokens=512):
    # 1. Extract text
    text = extract_pdf_text_fast(pdf_path)
    if not text:
        print("[ERROR] No text extracted from PDF. Aborting.")
        return {}

    # 2. Chunk text
    chunks = chunk_text(text, max_chars=chunk_size)
    print(f"PDF split into {len(chunks)} chunks")

    # 3. Extract KDEs per chunk
    kde_dicts = []
    for i, chunk in enumerate(chunks, start=1):
        print(f"Processing chunk {i}/{len(chunks)}...")
        try:
            # FIX #6: per-chunk error handling so one bad chunk doesn't abort the run
            prompt = prompt_builder(chunk)
            kde_output = run_llm(pipe, prompt, max_new_tokens=max_new_tokens)
            print(kde_output)
            parsed = parse_kde_yaml(kde_output)
            print(parsed)
            if parsed:
                kde_dicts.append(parsed)
        except Exception as e:
            print(f"[ERROR] Chunk {i} failed: {e}. Skipping.")

    # 4. Merge dictionaries in Python
    final_kdes = merge_kde_dicts(kde_dicts)
    return final_kdes




# ---------------------------------------------------------
# 9. MULTI-FILE PIPELINE: run all prompts, save per-file YAML
# ---------------------------------------------------------

ALL_PROMPT_STRATEGIES = [
    ("zero_shot",        build_kde_prompt_zero_shot),
    ("one_shot",         build_kde_prompt_one_shot),
    ("chain_of_thought", build_kde_prompt_chain_of_thought),
]


def process_two_files(pdf_path_1, pdf_path_2, pipe, output_dir=".", chunk_size=4000, max_new_tokens=512):
    """
    Run all three prompt strategies on each of two PDF files.
    KDEs from all strategies are merged and saved to one YAML file per input PDF.

    Args:
        pdf_path_1:     Path to the first PDF.
        pdf_path_2:     Path to the second PDF.
        pipe:           A loaded HuggingFace text-generation pipeline.
        output_dir:     Directory where the two output YAML files are written.
        chunk_size:     Max characters per text chunk fed to the model.
        max_new_tokens: Token budget for each model call.

    Returns:
        dict: { pdf_path: output_yaml_path } for each input file.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    results = {}

    for idx, pdf_path in enumerate((pdf_path_1, pdf_path_2), start=1):
        print(f"\n{'='*60}")
        print(f"Processing input{idx}: {pdf_path}")
        print(f"{'='*60}")

        all_strategy_dicts = []

        for strategy_name, prompt_builder in ALL_PROMPT_STRATEGIES:
            print(f"\n  -- Strategy: {strategy_name}")
            kdes = extract_kdes_from_pdf(
                pdf_path,
                pipe,
                prompt_builder,
                chunk_size=chunk_size,
                max_new_tokens=max_new_tokens,
            )
            if kdes:
                all_strategy_dicts.append(kdes)
            else:
                print(f"  [WARN] Strategy '{strategy_name}' produced no KDEs for {pdf_path}.")

        # Merge KDEs from all strategies into one flat dict
        combined = merge_kde_dicts(all_strategy_dicts)
        print(f"\n  Total KDEs extracted: {len(combined)}")

        # Prefix with input{{idx}} so duplicate paths never collide:
        # e.g. cis-r1.pdf -> input1-cis-r1_kdes.yaml / input2-cis-r1_kdes.yaml
        stem = os.path.splitext(os.path.basename(pdf_path))[0]
        output_path = os.path.join(output_dir, f"input{idx}-{stem}_kdes.yaml")

        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(combined, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

        print(f"  Saved -> {output_path}")
        results[f"input{idx}:{pdf_path}"] = output_path

    return results


# ---------------------------------------------------------
# 10. ENTRY POINT
# ---------------------------------------------------------
if __name__ == "__main__":
    print("Loading model...")
    pipe = pipeline(
        "text-generation",
        model="google/gemma-3-1b-it",
        device="cpu",
        dtype=torch.bfloat16,
    )
    print("Model loaded.\n")

    output_map = process_two_files(
        pdf_path_1="cis-r1.pdf",
        pdf_path_2="cis-r2.pdf",
        pipe=pipe,
        output_dir="kde_outputs",
    )

    print("\nOutput files:")
    for pdf, yaml_path in output_map.items():
        print(f"  {pdf} → {yaml_path}")