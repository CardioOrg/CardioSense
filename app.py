"""
CardioSense — Flask application entry-point.
Run:  python app.py
"""

import logging
from flask import (
    Flask, render_template, redirect, url_for, session, request, flash, g
)
import config

# ───────────────────────────── Logging ──────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ───────────────────────────── App factory ──────────────────────────
def create_app():
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )
    app.secret_key = config.SECRET_KEY
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH

    # ── Lazy-init Firebase on first request ───────────────────────
    @app.before_request
    def _init_firebase():
        from services.firebase_service import init_firebase
        init_firebase()

    # ── Inject user info into every template context ─────────────
    @app.before_request
    def _load_user():
        g.user = None
        if "user" in session:
            g.user = session["user"]

    @app.context_processor
    def _inject_user():
        return dict(current_user=g.get("user"), config=config)

    # ── Register blueprints ──────────────────────────────────────
    from routes.auth_routes import auth_bp
    from routes.patient_routes import patient_bp
    from routes.doctor_routes import doctor_bp
    from routes.component_routes import component_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(patient_bp, url_prefix="/patient")
    app.register_blueprint(doctor_bp, url_prefix="/doctor")
    app.register_blueprint(component_bp, url_prefix="/component")

    # ── Landing page ─────────────────────────────────────────────
    @app.route("/")
    def landing():
        if g.user:
            if g.user.get("role") == "doctor":
                return redirect(url_for("doctor.dashboard"))
            return redirect(url_for("patient.dashboard"))
        return render_template("landing.html")

    # ── Error handlers ───────────────────────────────────────────
    @app.errorhandler(401)
    def unauthorized(e):
        logger.warning("401 Unauthorized: %s", request.path)
        flash("Please log in to continue.", "warning")
        return redirect(url_for("auth.login"))

    @app.errorhandler(403)
    def forbidden(e):
        logger.warning("403 Forbidden: %s", request.path)
        flash("You do not have permission to access this page.", "danger")
        return redirect(url_for("landing"))

    @app.errorhandler(404)
    def not_found(e):
        logger.info("404 Not Found: %s", request.path)
        return render_template("base.html", error_title="404",
                               error_message="Page not found."), 404

    @app.errorhandler(500)
    def server_error(e):
        logger.exception("500 Internal Server Error: %s", request.path)
        return render_template("base.html", error_title="500",
                               error_message="An internal error occurred."), 500

    logger.info("CardioSense app created successfully.")
    return app


# ──────────────────────────── Run locally ──────────────────────────
if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, host="0.0.0.0", port=5000)
