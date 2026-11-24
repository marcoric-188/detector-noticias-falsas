"""Estilo común de las gráficas: marcas finas, ejes discretos, títulos y unidades siempre presentes."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import config  # noqa: E402

#: Paleta categórica validada (azul, naranja, aqua) y tintas de texto.
AZUL, NARANJA, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
TEXTO, TEXTO_2, REJILLA = "#0b0b0b", "#52514e", "#e4e3df"


def estilo() -> None:
    """Aplica el estilo de matplotlib del proyecto."""
    plt.rcParams.update({
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
        "axes.edgecolor": REJILLA, "axes.labelcolor": TEXTO_2, "xtick.color": TEXTO_2,
        "ytick.color": TEXTO_2, "text.color": TEXTO, "axes.titlecolor": TEXTO,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": REJILLA, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    })


def guardar(fig, nombre: str) -> None:
    """Guarda la figura en ``results/figuras/`` a 150 ppp y la cierra."""
    destino = config.ruta(config.base()["salidas"]["dir_figuras"])
    destino.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(destino / nombre, dpi=150)
    plt.close(fig)
