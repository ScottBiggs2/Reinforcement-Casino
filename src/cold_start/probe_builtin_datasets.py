"""Builtin linguistic / factual probe corpora for linear probes on MLP hooks.

Sourced from a prior integration branch's probe-analysis utilities and kept as a
standalone module so ``mask_probe_report`` and ``src/analysis/probe_analysis.py``
can share one definition of ``PROBE_DATASETS``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

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


def layer_index_from_hook_name(layer_name: str) -> int:
    """Decoder layer index from hook names like ``model.layers.12.mlp.down_proj``."""
    parts = str(layer_name).split(".")
    for i, part in enumerate(parts):
        if part == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                pass
    return 0


def validate_probe_datasets() -> None:
    """Assert balanced class counts and equal example counts across properties."""
    expected_n: Optional[int] = None
    for prop_name, prop_data in PROBE_DATASETS.items():
        examples = prop_data["examples"]
        n = len(examples)
        if expected_n is None:
            expected_n = n
        if n != expected_n:
            raise ValueError(f"Property {prop_name!r} has {n} examples; expected {expected_n}")
        labels = [lbl for _, lbl in examples]
        n_pos = sum(1 for l in labels if l == 1)
        n_neg = sum(1 for l in labels if l == 0)
        if n_pos != n_neg:
            raise ValueError(f"Property {prop_name!r} imbalanced: {n_pos} pos vs {n_neg} neg")


def build_concatenated_texts_and_slices(
    property_names: Sequence[str],
) -> Tuple[List[str], Dict[str, slice], Dict[str, np.ndarray]]:
    """Concatenate builtin texts in ``property_names`` order; return slices + label arrays."""
    all_texts: List[str] = []
    property_slices: Dict[str, slice] = {}
    labels_by_prop: Dict[str, np.ndarray] = {}
    for prop_name in property_names:
        if prop_name not in PROBE_DATASETS:
            raise KeyError(f"Unknown probe property {prop_name!r}. Keys: {list(PROBE_DATASETS)}")
        prop_data = PROBE_DATASETS[prop_name]
        texts = [t for t, _ in prop_data["examples"]]
        property_slices[prop_name] = slice(len(all_texts), len(all_texts) + len(texts))
        all_texts.extend(texts)
        labels_by_prop[prop_name] = np.array([lbl for _, lbl in prop_data["examples"]], dtype=np.int64)
    return all_texts, property_slices, labels_by_prop


def train_linear_probes_cv(
    activations_by_layer: Dict[str, torch.Tensor],
    labels: np.ndarray,
    *,
    cv: int = 5,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Train one L2 logistic probe per layer; return (mean_cv_by_layer, diagnostics).

    Uses StratifiedKFold (same spirit as prior probe-analysis utilities).
    """
    results: Dict[str, float] = {}
    skipped: List[str] = []
    n = len(labels)
    if n < 4:
        return results, {"error": "too_few_samples", "n": n}

    uniq, counts = np.unique(labels, return_counts=True)
    if len(uniq) < 2:
        return results, {"error": "single_class_labels"}

    min_class = int(counts.min())
    n_splits = min(int(cv), min_class, max(2, n // 2))
    n_splits = max(2, n_splits)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for layer_name, acts in activations_by_layer.items():
        X = acts.float().numpy()
        if X.shape[0] != len(labels):
            skipped.append(layer_name)
            continue
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X)
        clf = LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            C=1.0,
            max_iter=500,
            random_state=42,
        )
        try:
            scores = cross_val_score(clf, X_sc, labels, cv=skf, scoring="accuracy")
            results[layer_name] = float(scores.mean())
        except ValueError:
            skipped.append(layer_name)
    diag = {
        "n_layers_scored": len(results),
        "skipped_layers": skipped,
        "cv_folds_requested": cv,
        "cv_folds_effective": n_splits,
    }
    return results, diag


def summarize_layer_scores(layer_scores: Dict[str, float]) -> Dict[str, Optional[float]]:
    vals = [v for v in layer_scores.values() if v == v]
    if not vals:
        return {"mean_cv_accuracy": None, "min_cv_accuracy": None, "max_cv_accuracy": None, "n_layers": 0}
    return {
        "mean_cv_accuracy": float(sum(vals) / len(vals)),
        "min_cv_accuracy": float(min(vals)),
        "max_cv_accuracy": float(max(vals)),
        "n_layers": len(vals),
    }
