from flask import Flask, request, jsonify, render_template
from Models.utapis_detector import UtapisDetector

app = Flask(__name__, template_folder="Pages", static_folder="static")
detector = UtapisDetector()


@app.route("/", methods=["GET"])
def index():
    """
    Halaman utama: form input paragraf.
    """
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect_form():
    """
    Endpoint untuk form HTML (UI).
    Ambil paragraf dari textarea, jalankan UTAPIS,
    lalu render result.html.
    """
    paragraph = request.form.get("paragraph", "") or ""
    results = detector.detect_particles(paragraph)

    # Kirim ke template:
    # - paragraph: teks asli
    # - results: list dict hasil UTAPIS
    return render_template(
        "result.html",
        paragraph=paragraph,
        results=results,
    )


@app.route("/detect-utapis", methods=["POST"])
def detect_utapis():
    """
    Endpoint API JSON (untuk front-end lain / integrasi).
    Body: { "paragraph": "..." }
    """
    data = request.get_json(silent=True) or {}
    paragraph = data.get("paragraph", "") or ""
    results = detector.detect_particles(paragraph)

    return jsonify({
        "paragraph": paragraph,
        "results": results,
    })


if __name__ == "__main__":
    app.run(debug=True)
