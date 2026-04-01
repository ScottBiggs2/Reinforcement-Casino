#!/usr/bin/env python3
"""
Probing classifier analysis for sparse mask comparison.

For each mask configuration, extracts per-layer MLP activations then trains
a linear probe per layer on several linguistic/cognitive properties.
Produces a heatmap showing which mask retains which type of knowledge at
each layer — analogous to the probing literature (e.g. Tenney et al. 2019).

Usage:
    python src/analysis/probe_analysis.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --mask_a masks/cold_snip_llama_90pct.pt \
        --mask_b masks/warm_momentum_llama_90pct.pt \
        --mask_a_label "SNIP" \
        --mask_b_label "Momentum" \
        --output_dir probe_results/

Optional flags:
    --include_baseline      also evaluate the unmasked model
    --layer_stride N        sample every N-th layer (default: 4)
    --batch_size N          inference batch size (default: 8)
    --max_length N          token length cap (default: 128)
"""

import argparse
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from cold_start.utils.activation_hooks import FeatureExtractor


# ---------------------------------------------------------------------------
# Built-in probe datasets  (30 per class = 60 total per property)
# Labels: 0 = class A, 1 = class B
# All properties use distinct text corpora; texts are concatenated for a
# single inference pass per config, then sliced back by property.
# ---------------------------------------------------------------------------

PROBE_DATASETS = {
    "syntax": {
        "description": "Subject-verb number agreement (0=singular, 1=plural)",
        "examples": [
            # singular
            ("The dog runs in the park every morning.", 0),
            ("A child plays with the colorful toys.", 0),
            ("The scientist writes a detailed report.", 0),
            ("My neighbor walks his dog after dinner.", 0),
            ("The student reads the textbook carefully.", 0),
            ("A teacher explains the difficult concept.", 0),
            ("The bird sings a beautiful melody.", 0),
            ("One athlete trains hard every single day.", 0),
            ("The manager reviews every employee's work.", 0),
            ("A doctor examines patients in the hospital.", 0),
            ("The programmer writes clean, readable code.", 0),
            ("Every citizen votes in the local election.", 0),
            ("The chef prepares fresh ingredients daily.", 0),
            ("A cat sits quietly on the warm windowsill.", 0),
            ("The engineer designs efficient algorithms.", 0),
            ("The journalist reports breaking news every hour.", 0),
            ("A lawyer defends clients in the courtroom.", 0),
            ("The pilot navigates the aircraft carefully.", 0),
            ("One musician practices scales for two hours.", 0),
            ("The librarian organizes books on the shelves.", 0),
            ("A firefighter rescues people from burning buildings.", 0),
            ("The carpenter builds sturdy furniture from oak.", 0),
            ("One gardener tends flowers in the greenhouse.", 0),
            ("The accountant reviews the annual budget report.", 0),
            ("A nurse checks the patient's vital signs hourly.", 0),
            ("The architect designs modern residential buildings.", 0),
            ("One translator converts documents into French.", 0),
            ("The photographer captures stunning landscape images.", 0),
            ("A sailor navigates the vessel through rough waters.", 0),
            ("The electrician installs wiring in new constructions.", 0),
            # plural
            ("The dogs run in the park every morning.", 1),
            ("Several children play with the colorful toys.", 1),
            ("The scientists write detailed reports together.", 1),
            ("My neighbors walk their dogs after dinner.", 1),
            ("The students read their textbooks carefully.", 1),
            ("The teachers explain the difficult concepts.", 1),
            ("The birds sing beautiful melodies at dawn.", 1),
            ("Many athletes train hard every single day.", 1),
            ("The managers review every employee's work.", 1),
            ("Several doctors examine patients in the hospital.", 1),
            ("The programmers write clean, readable code.", 1),
            ("All citizens vote in the local election.", 1),
            ("The chefs prepare fresh ingredients daily.", 1),
            ("Some cats sit quietly on the warm windowsills.", 1),
            ("The engineers design efficient algorithms.", 1),
            ("The journalists report breaking news every hour.", 1),
            ("Several lawyers defend clients in the courtroom.", 1),
            ("The pilots navigate their aircraft carefully.", 1),
            ("Many musicians practice scales for two hours.", 1),
            ("The librarians organize books on the shelves.", 1),
            ("Several firefighters rescue people from burning buildings.", 1),
            ("The carpenters build sturdy furniture from oak.", 1),
            ("Many gardeners tend flowers in the greenhouse.", 1),
            ("The accountants review the annual budget reports.", 1),
            ("Several nurses check the patients' vital signs hourly.", 1),
            ("The architects design modern residential buildings.", 1),
            ("Many translators convert documents into French.", 1),
            ("The photographers capture stunning landscape images.", 1),
            ("Several sailors navigate their vessels through rough waters.", 1),
            ("The electricians install wiring in new constructions.", 1),
        ],
    },
    "semantics": {
        "description": "Sentiment polarity (0=negative, 1=positive)",
        "examples": [
            # negative
            ("This film was absolutely terrible and a complete waste of time.", 0),
            ("I hate how the traffic makes me late every single day.", 0),
            ("The food at that restaurant was disgusting and cold.", 0),
            ("What a deeply disappointing experience at the customer service desk.", 0),
            ("The weather is dreadful and ruins all my outdoor plans.", 0),
            ("That was the worst performance I have ever had to sit through.", 0),
            ("The product broke immediately and the company ignored my complaint.", 0),
            ("I deeply regret spending money on this completely useless software.", 0),
            ("The meeting was a frustrating and utterly unproductive waste of hours.", 0),
            ("The staff were rude and unhelpful at every turn.", 0),
            ("Nothing about this project went well from start to finish.", 0),
            ("I feel terrible after the exhausting and miserable commute.", 0),
            ("The lecture was boring and the instructor was impossible to follow.", 0),
            ("The hotel room was filthy and the staff were unfriendly.", 0),
            ("This is a poorly designed and deeply frustrating application.", 0),
            ("The customer support was shockingly unhelpful and dismissive.", 0),
            ("I wasted an entire afternoon on this broken and confusing tool.", 0),
            ("The concert was a huge letdown with terrible sound quality.", 0),
            ("Working at this company has been nothing but stress and disappointment.", 0),
            ("The book was so dull I could barely finish the first chapter.", 0),
            ("The packaging was damaged, the item was wrong, and the return process was awful.", 0),
            ("I cannot believe how poorly this event was organized.", 0),
            ("The interface is cluttered, slow, and makes no sense at all.", 0),
            ("Every interaction with this team has been frustrating and unpleasant.", 0),
            ("The renovation left the house looking worse than before.", 0),
            ("This trip has been one disappointment after another.", 0),
            ("The instructions were confusing and the assembly was a nightmare.", 0),
            ("I regret every minute I spent trying to get this to work.", 0),
            ("The system crashes constantly and the developers seem indifferent.", 0),
            ("This experience was so negative it has put me off entirely.", 0),
            # positive
            ("This film was absolutely wonderful and a genuine joy to watch.", 1),
            ("I love how the sunshine brightens up the entire day.", 1),
            ("The food at that restaurant was delicious and beautifully presented.", 1),
            ("What a fantastic and memorable experience at the customer service desk.", 1),
            ("The weather is gorgeous and perfect for outdoor activities today.", 1),
            ("That was the best performance I have ever had the pleasure to witness.", 1),
            ("The product has exceeded all my expectations and works perfectly.", 1),
            ("I am absolutely thrilled with this incredibly useful and polished software.", 1),
            ("The meeting was productive and full of great collaborative energy.", 1),
            ("The staff were warm, friendly, and helpful at every turn.", 1),
            ("Everything about this project went smoothly from start to finish.", 1),
            ("I feel energized and refreshed after the wonderful morning walk.", 1),
            ("The lecture was engaging and the instructor was very clear and inspiring.", 1),
            ("The hotel room was spotless and the staff were exceptionally kind.", 1),
            ("This is a delightful and beautifully designed application.", 1),
            ("The customer support team was incredibly responsive and knowledgeable.", 1),
            ("I am so glad I found this tool — it has saved me countless hours.", 1),
            ("The concert was an unforgettable experience with outstanding sound quality.", 1),
            ("Working at this company has been a true pleasure and a great opportunity.", 1),
            ("The book was so captivating I finished it in a single sitting.", 1),
            ("The packaging was perfect, the item arrived early, and it was exactly as described.", 1),
            ("I was amazed by how smoothly and professionally this event was organized.", 1),
            ("The interface is clean, fast, and incredibly intuitive to use.", 1),
            ("Every interaction with this team has been positive and inspiring.", 1),
            ("The renovation transformed the house into something truly beautiful.", 1),
            ("This trip has been one delightful surprise after another.", 1),
            ("The instructions were crystal clear and assembly was a breeze.", 1),
            ("I am genuinely impressed by how well this works right out of the box.", 1),
            ("The system is stable, fast, and the developers are very responsive.", 1),
            ("This experience has been so positive it has become my go-to choice.", 1),
        ],
    },
    "factual": {
        "description": "Entity type: scientist/person (0) vs geographic location (1)",
        "examples": [
            # persons / scientists
            ("Albert Einstein developed the theory of general relativity in 1915.", 0),
            ("Marie Curie was the first woman to win a Nobel Prize.", 0),
            ("Isaac Newton formulated the laws of motion and universal gravitation.", 0),
            ("Nikola Tesla invented the alternating current electrical system.", 0),
            ("Ada Lovelace wrote the first algorithm intended for a computing machine.", 0),
            ("Charles Darwin proposed the theory of natural selection in 1859.", 0),
            ("Rosalind Franklin produced crucial X-ray diffraction data for DNA structure.", 0),
            ("Alan Turing laid the theoretical foundations of modern computer science.", 0),
            ("Richard Feynman made fundamental contributions to quantum electrodynamics.", 0),
            ("Florence Nightingale pioneered modern nursing and hospital sanitation practices.", 0),
            ("Galileo Galilei improved the telescope and supported the heliocentric model.", 0),
            ("Leonhard Euler made major contributions to mathematics, physics, and astronomy.", 0),
            ("Emmy Noether developed abstract algebra and proved Noether's theorem.", 0),
            ("Stephen Hawking worked on black holes, Hawking radiation, and cosmological singularities.", 0),
            ("Linus Torvalds created the Linux kernel and released it publicly in 1991.", 0),
            ("James Clerk Maxwell unified electricity, magnetism, and light into one theory.", 0),
            ("Dmitri Mendeleev created the periodic table of chemical elements.", 0),
            ("Niels Bohr developed the first quantum model of the hydrogen atom.", 0),
            ("Grace Hopper invented one of the first compiler tools for programming languages.", 0),
            ("Louis Pasteur developed germ theory and created the first vaccines for rabies.", 0),
            ("Werner Heisenberg formulated the uncertainty principle in quantum mechanics.", 0),
            ("Max Planck introduced the concept of energy quanta, founding quantum theory.", 0),
            ("Carl Sagan popularized astronomy and the search for extraterrestrial intelligence.", 0),
            ("Barbara McClintock discovered that genes can change positions on chromosomes.", 0),
            ("Erwin Schrödinger formulated the wave equation central to quantum mechanics.", 0),
            ("Paul Dirac predicted the existence of antimatter through his relativistic equation.", 0),
            ("John von Neumann developed the architecture used by most modern computers.", 0),
            ("James Watson and Francis Crick proposed the double helix structure of DNA.", 0),
            ("Enrico Fermi built the first nuclear reactor and worked on the Manhattan Project.", 0),
            ("Hedy Lamarr co-invented frequency-hopping spread spectrum communication technology.", 0),
            # geographic locations
            ("The Amazon River flows through the heart of South America into the Atlantic Ocean.", 1),
            ("The Sahara Desert covers most of northern Africa and is the world's largest hot desert.", 1),
            ("The Himalayas contain the world's highest mountain peaks, including Mount Everest.", 1),
            ("The Mediterranean Sea is bordered by Europe, Africa, and the Middle East.", 1),
            ("The Great Barrier Reef lies off the northeastern coast of Queensland, Australia.", 1),
            ("The Nile River is traditionally considered the world's longest river.", 1),
            ("The Arctic Ocean surrounds the North Pole and is covered by sea ice year-round.", 1),
            ("The Grand Canyon was carved over millions of years by the Colorado River.", 1),
            ("The Gobi Desert stretches across northern China and southern Mongolia.", 1),
            ("The Pacific Ocean is the largest and deepest ocean on Earth.", 1),
            ("The Congo Basin is home to the world's second-largest tropical rainforest.", 1),
            ("The Alps run across France, Switzerland, Italy, Germany, and Austria.", 1),
            ("The Mariana Trench in the western Pacific is the deepest point in any ocean.", 1),
            ("Lake Baikal in Siberia holds approximately twenty percent of the world's fresh surface water.", 1),
            ("The Atacama Desert in South America is one of the driest places on Earth.", 1),
            ("The Serengeti plains in Tanzania are famous for the annual wildebeest migration.", 1),
            ("The Tibetan Plateau is often called the roof of the world due to its high elevation.", 1),
            ("The Dead Sea, bordered by Jordan and Israel, is the lowest point on Earth's surface.", 1),
            ("The Great Rift Valley stretches from Lebanon in the north to Mozambique in the south.", 1),
            ("The Mekong River flows through China, Myanmar, Laos, Thailand, Cambodia, and Vietnam.", 1),
            ("The Andes Mountains run along the entire western coast of South America.", 1),
            ("The Okavango Delta in Botswana is one of the world's largest inland deltas.", 1),
            ("The Bering Strait separates Asia from North America by only 85 kilometers.", 1),
            ("The Danube River flows through ten countries before emptying into the Black Sea.", 1),
            ("The Namib Desert along Africa's southwest coast is one of the oldest deserts on Earth.", 1),
            ("The Pantanal in South America is the world's largest tropical wetland area.", 1),
            ("The Sea of Japan lies between the Japanese archipelago and the Asian continent.", 1),
            ("The Scottish Highlands contain some of the oldest exposed rock formations in Europe.", 1),
            ("The Patagonian Steppe in southern Argentina is one of the windiest places on Earth.", 1),
            ("The Aral Sea, once one of the world's largest lakes, has largely dried up since the 1960s.", 1),
        ],
    },
    "math": {
        "description": "Arithmetic correctness (0=false, 1=true)",
        "examples": [
            # false
            ("The result of three plus five is nine.", 0),
            ("Multiplying six by seven gives forty-five.", 0),
            ("The square root of sixteen is five.", 0),
            ("Twelve divided by four equals four.", 0),
            ("Two raised to the power of five equals thirty.", 0),
            ("The product of eight and nine is seventy-one.", 0),
            ("Fifteen minus eight equals six.", 0),
            ("The factorial of four equals twenty-two.", 0),
            ("One hundred divided by five is twenty-one.", 0),
            ("Adding thirteen and nineteen gives thirty-three.", 0),
            ("Nine times six is fifty-two.", 0),
            ("The cube of three equals twenty-six.", 0),
            ("Fifty minus seventeen is thirty-two.", 0),
            ("Four multiplied by twelve is forty-six.", 0),
            ("The sum of seven, eight, and nine is twenty-three.", 0),
            ("Twenty-five percent of eighty is fifteen.", 0),
            ("The square of eleven is one hundred and ten.", 0),
            ("Sixty divided by four is thirteen.", 0),
            ("Three to the power of four equals seventy-nine.", 0),
            ("The greatest common divisor of twelve and eighteen is five.", 0),
            ("Forty-eight divided by six is nine.", 0),
            ("The sum of the first five natural numbers is fourteen.", 0),
            ("Two hundred minus seventy-three equals one hundred and twenty-six.", 0),
            ("Seven squared is forty-six.", 0),
            ("The product of five, four, and three is fifty-seven.", 0),
            ("Thirty percent of ninety is twenty-five.", 0),
            ("The remainder when seventeen is divided by five is one.", 0),
            ("Adding sixty-four and thirty-seven gives ninety-eight.", 0),
            ("The cube root of twenty-seven is four.", 0),
            ("Eleven times eleven equals one hundred and twenty.", 0),
            # true
            ("The result of three plus five is eight.", 1),
            ("Multiplying six by seven gives forty-two.", 1),
            ("The square root of sixteen is four.", 1),
            ("Twelve divided by four equals three.", 1),
            ("Two raised to the power of five equals thirty-two.", 1),
            ("The product of eight and nine is seventy-two.", 1),
            ("Fifteen minus eight equals seven.", 1),
            ("The factorial of four equals twenty-four.", 1),
            ("One hundred divided by five is twenty.", 1),
            ("Adding thirteen and nineteen gives thirty-two.", 1),
            ("Nine times six is fifty-four.", 1),
            ("The cube of three equals twenty-seven.", 1),
            ("Fifty minus seventeen is thirty-three.", 1),
            ("Four multiplied by twelve is forty-eight.", 1),
            ("The sum of seven, eight, and nine is twenty-four.", 1),
            ("Twenty-five percent of eighty is twenty.", 1),
            ("The square of eleven is one hundred and twenty-one.", 1),
            ("Sixty divided by four is fifteen.", 1),
            ("Three to the power of four equals eighty-one.", 1),
            ("The greatest common divisor of twelve and eighteen is six.", 1),
            ("Forty-eight divided by six is eight.", 1),
            ("The sum of the first five natural numbers is fifteen.", 1),
            ("Two hundred minus seventy-three equals one hundred and twenty-seven.", 1),
            ("Seven squared is forty-nine.", 1),
            ("The product of five, four, and three is sixty.", 1),
            ("Thirty percent of ninety is twenty-seven.", 1),
            ("The remainder when seventeen is divided by five is two.", 1),
            ("Adding sixty-four and thirty-seven gives one hundred and one.", 1),
            ("The cube root of twenty-seven is three.", 1),
            ("Eleven times eleven equals one hundred and twenty-one.", 1),
        ],
    },
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _layer_index(layer_name: str) -> int:
    """Extract integer layer index from names like 'model.layers.12.mlp.down_proj'."""
    parts = layer_name.split(".")
    for i, part in enumerate(parts):
        if part == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                pass
    return 0


def load_mask(path: str) -> dict:
    """Load a mask file, handling both raw dicts and wrapped {masks: ...} format."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(data, dict) and "masks" in data:
        return data["masks"]
    return data


@contextmanager
def apply_mask(model, mask_dict: dict):
    """Context manager: temporarily zero out weights according to mask_dict.

    mask_dict maps param_name -> binary tensor (1=keep, 0=prune).
    Original weights are restored on exit regardless of exceptions.
    """
    originals = {}
    try:
        for name, param in model.named_parameters():
            if name in mask_dict:
                originals[name] = param.data.clone()
                param.data.mul_(mask_dict[name].to(param.device, dtype=param.dtype))
        yield
    finally:
        for name, param in model.named_parameters():
            if name in originals:
                param.data.copy_(originals[name])


@contextmanager
def no_mask():
    """Dummy context manager for the unmasked baseline."""
    yield


# ---------------------------------------------------------------------------
# Probe evaluation
# ---------------------------------------------------------------------------

def train_probes(activations_by_layer: dict, labels: np.ndarray, cv: int = 5) -> dict:
    """Train a linear probe per layer and return cross-validated accuracy.

    Args:
        activations_by_layer: {layer_name: Tensor[N, D]}
        labels: np.array of shape [N] with binary labels
        cv: number of cross-validation folds

    Returns:
        {layer_name: float mean_cv_accuracy}
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    results = {}

    for layer_name, acts in activations_by_layer.items():
        X = acts.float().numpy()
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X)

        clf = LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            C=1.0,
            max_iter=500,
            random_state=42,
        )
        scores = cross_val_score(clf, X_sc, labels, cv=skf, scoring="accuracy")
        results[layer_name] = float(scores.mean())

    return results


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_probe_heatmap(
    results_by_config: dict,
    sample_indices: list,
    property_order: list,
    output_path: str,
):
    """Plot a grid of heatmaps: one panel per mask config.

    Args:
        results_by_config: {config_label: {prop_name: {layer_name: accuracy}}}
        sample_indices: list of integer layer indices that were sampled
        property_order: list of property names in display order
        output_path: path for the output PNG
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches

    n_configs = len(results_by_config)
    n_props = len(property_order)
    n_layers = len(sample_indices)

    layer_labels = [f"layer {i}" for i in sample_indices]
    # Colormap: chance (0.5) → yellow, 1.0 → dark green; below chance → red
    cmap = plt.cm.RdYlGn
    vmin, vmax = 0.5, 1.0

    fig_width = max(10, 4 * n_configs + 2)
    fig, axes = plt.subplots(
        1, n_configs,
        figsize=(fig_width, n_props * 0.9 + 2.5),
        squeeze=False,
    )
    axes = axes[0]

    for ax, (config_label, prop_results) in zip(axes, results_by_config.items()):
        # Build [n_props, n_layers] accuracy matrix
        mat = np.full((n_props, n_layers), np.nan)
        for pi, prop in enumerate(property_order):
            if prop not in prop_results:
                continue
            layer_map = prop_results[prop]
            sorted_layer_names = sorted(layer_map.keys(), key=_layer_index)
            for li, lname in enumerate(sorted_layer_names):
                if li < n_layers:
                    mat[pi, li] = layer_map[lname]

        # Clamp values below chance to vmin for display purposes
        display_mat = np.where(np.isnan(mat), vmin, np.clip(mat, vmin, vmax))

        im = ax.imshow(display_mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

        ax.set_xticks(range(n_layers))
        ax.set_xticklabels(layer_labels, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(n_props))
        ax.set_yticklabels(property_order, fontsize=11)
        ax.set_title(config_label, fontsize=13, fontweight="bold", pad=10)

        # Annotate cells
        for pi in range(n_props):
            for li in range(n_layers):
                val = mat[pi, li]
                if np.isnan(val):
                    continue
                txt_color = "white" if val < 0.6 or val > 0.88 else "black"
                ax.text(
                    li, pi, f"{val:.2f}",
                    ha="center", va="center",
                    fontsize=7.5, color=txt_color, fontweight="bold",
                )

    # Shared colorbar
    fig.subplots_adjust(right=0.86, wspace=0.35)
    cbar_ax = fig.add_axes([0.89, 0.15, 0.025, 0.7])
    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=mcolors.Normalize(vmin=vmin, vmax=vmax),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("probe accuracy", fontsize=10, labelpad=8)
    cbar.set_ticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    cbar_ax.text(
        0.5, -0.04, "low\n(knowledge lost)",
        ha="center", va="top", transform=cbar_ax.transAxes, fontsize=8,
    )
    cbar_ax.text(
        0.5, 1.04, "high\n(knowledge retained)",
        ha="center", va="bottom", transform=cbar_ax.transAxes, fontsize=8,
    )

    fig.suptitle("Probing Classifier Analysis: Knowledge Retention per Layer", fontsize=14, y=1.01)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"[plot] Saved heatmap → {output_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Probing classifier analysis for mask comparison")
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct",
                   help="HuggingFace model name or local path")
    p.add_argument("--mask_a", required=True, help="Path to mask A .pt file")
    p.add_argument("--mask_b", required=True, help="Path to mask B .pt file")
    p.add_argument("--mask_a_label", default="Mask A", help="Display label for mask A")
    p.add_argument("--mask_b_label", default="Mask B", help="Display label for mask B")
    p.add_argument("--include_baseline", action="store_true",
                   help="Also run on the unmasked model as a baseline panel")
    p.add_argument("--output_dir", default="probe_results",
                   help="Directory for JSON and PNG outputs")
    p.add_argument("--layer_stride", type=int, default=4,
                   help="Plot every N-th layer (default: 4). Use 1 for all layers.")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--cv_folds", type=int, default=5,
                   help="Cross-validation folds for probe accuracy (default: 5)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Validate probe datasets
    # ------------------------------------------------------------------
    expected_n = None
    for prop_name, prop_data in PROBE_DATASETS.items():
        n = len(prop_data["examples"])
        if expected_n is None:
            expected_n = n
        assert n == expected_n, (
            f"Property '{prop_name}' has {n} examples; expected {expected_n}"
        )
        labels = [lbl for _, lbl in prop_data["examples"]]
        n_pos = sum(1 for l in labels if l == 1)
        n_neg = sum(1 for l in labels if l == 0)
        assert n_pos == n_neg, (
            f"Property '{prop_name}' is imbalanced: {n_pos} pos vs {n_neg} neg"
        )

    # ------------------------------------------------------------------
    # Concatenate all texts for a single inference pass
    # ------------------------------------------------------------------
    all_texts = []
    property_slices = {}
    for prop_name, prop_data in PROBE_DATASETS.items():
        texts = [t for t, _ in prop_data["examples"]]
        property_slices[prop_name] = slice(len(all_texts), len(all_texts) + len(texts))
        all_texts.extend(texts)

    print(f"[main] {len(PROBE_DATASETS)} probe properties, "
          f"{expected_n} examples each, {len(all_texts)} texts total")

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print(f"\n[main] Loading tokenizer & model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    # Count MLP layers and build sample index list
    n_mlp_layers = sum(
        1 for name, _ in model.named_modules() if name.endswith("down_proj")
    )
    sample_indices = list(range(0, n_mlp_layers, args.layer_stride))
    if (n_mlp_layers - 1) not in sample_indices:
        sample_indices.append(n_mlp_layers - 1)
    sample_index_set = set(sample_indices)
    print(f"[main] Model has {n_mlp_layers} MLP layers; "
          f"sampling {len(sample_indices)} layers: {sample_indices}")

    # ------------------------------------------------------------------
    # Load masks
    # ------------------------------------------------------------------
    print(f"\n[main] Loading mask A: {args.mask_a}")
    mask_a = load_mask(args.mask_a)
    print(f"[main] Loading mask B: {args.mask_b}")
    mask_b = load_mask(args.mask_b)

    # ------------------------------------------------------------------
    # Build configs
    # ------------------------------------------------------------------
    configs = {}
    if args.include_baseline:
        configs["Baseline\n(no mask)"] = None
    configs[args.mask_a_label] = mask_a
    configs[args.mask_b_label] = mask_b

    # ------------------------------------------------------------------
    # Activation collection + probe training
    # ------------------------------------------------------------------
    extractor = FeatureExtractor().register(model)
    results_by_config = {}

    for config_label, mask_dict in configs.items():
        label_clean = config_label.replace("\n", " ")
        print(f"\n[main] ===== {label_clean} =====")

        ctx = apply_mask(model, mask_dict) if mask_dict is not None else no_mask()

        with ctx:
            device = next(model.parameters()).device
            activations = extractor.collect(
                model, tokenizer, all_texts, device,
                max_length=args.max_length,
                batch_size=args.batch_size,
            )

        # Filter to sampled layers only
        sampled_acts = {
            name: acts
            for name, acts in activations.items()
            if _layer_index(name) in sample_index_set
        }

        prop_results = {}
        for prop_name, prop_data in PROBE_DATASETS.items():
            slc = property_slices[prop_name]
            labels_arr = np.array([lbl for _, lbl in prop_data["examples"]])

            # Slice activations for this property's texts
            prop_acts = {name: acts[slc] for name, acts in sampled_acts.items()}

            layer_accs = train_probes(prop_acts, labels_arr, cv=args.cv_folds)
            prop_results[prop_name] = layer_accs

            vals = list(layer_accs.values())
            print(f"  {prop_name:12s}: mean={np.mean(vals):.3f}  "
                  f"min={np.min(vals):.3f}  max={np.max(vals):.3f}")

        results_by_config[config_label] = prop_results

    extractor.remove()

    # ------------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------------
    json_path = os.path.join(args.output_dir, "probe_results.json")
    with open(json_path, "w") as f:
        json.dump(results_by_config, f, indent=2)
    print(f"\n[main] Saved results JSON → {json_path}")

    # ------------------------------------------------------------------
    # Plot heatmap
    # ------------------------------------------------------------------
    plot_path = os.path.join(args.output_dir, "probe_heatmap.png")
    plot_probe_heatmap(
        results_by_config,
        sample_indices=sample_indices,
        property_order=list(PROBE_DATASETS.keys()),
        output_path=plot_path,
    )

    print("\n[main] Done.")


if __name__ == "__main__":
    main()
