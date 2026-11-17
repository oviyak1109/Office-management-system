
from pathlib import Path
from functools import wraps
import os
import re
import uuid
import joblib
from flask import Flask, render_template, request, redirect, url_for, session, flash, send_from_directory, abort
from werkzeug.utils import secure_filename
from pypdf import PdfReader
from sklearn.metrics.pairwise import cosine_similarity
from database import connect_database

BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "change-this-secret-key-before-deployment")
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024
UPLOAD_FOLDER = BASE_DIR / "uploads" / "resumes"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
MODEL_PATH = BASE_DIR / "train_model" / "resume_vectorizer.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError("Resume model not found. Run train_model.py first.")

resume_model = joblib.load(MODEL_PATH)


def login_required(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in first.", "error")
            return redirect(url_for("home"))
        return function(*args, **kwargs)
    return wrapper


def role_required(*roles):
    def decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            if "user_id" not in session:
                return redirect(url_for("home"))
            if session.get("role") not in roles:
                abort(403)
            return function(*args, **kwargs)
        return wrapper
    return decorator


def extract_pdf_text(pdf_path):
    reader = PdfReader(str(pdf_path))
    if reader.is_encrypted:
        raise ValueError("Password-protected PDFs are not supported.")
    text = ""
    for page in reader.pages:
        text += (page.extract_text() or "") + "\n"
    if not text.strip():
        raise ValueError("No readable text found in this PDF.")
    return text


def analyze_resume(resume_text, job_description, required_skills):
    skills = [s.strip() for s in required_skills.split(",") if s.strip()]
    matched = []
    missing = []
    for skill in skills:
        pattern = r"(?<!\w)" + re.escape(skill.lower()) + r"(?!\w)"
        if re.search(pattern, resume_text.lower()):
            matched.append(skill)
        else:
            missing.append(skill)
    skill_score = len(matched) / len(skills) * 100 if skills else 0
    resume_vector = resume_model.transform([resume_text])
    job_vector = resume_model.transform([job_description + " " + required_skills])
    similarity = cosine_similarity(resume_vector, job_vector)[0][0]
    text_similarity = max(0, min(100, similarity * 100))
    return {
        "match_score": round(skill_score * 0.70 + text_similarity * 0.30, 2),
        "text_similarity": round(text_similarity, 2),
        "matched_skills": matched,
        "missing_skills": missing
    }


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        login_id = request.form.get("login_id", "").strip()
        password = request.form.get("password", "")
        connection = connect_database()
        cursor = connection.cursor(dictionary=True)
        try:
            cursor.execute("SELECT * FROM users WHERE login_id = %s", (login_id,))
            user = cursor.fetchone()
        finally:
            cursor.close()
            connection.close()

        if user and user["password_hash"] == password:
            if not user.get("is_active", 1):
                flash("Your account is inactive. Contact the administrator.", "error")
                return redirect(url_for("home"))
            session["user_id"] = user["user_id"]
            session["login_id"] = user["login_id"]
            session["full_name"] = user["full_name"]
            session["role"] = user["role"]
            session["department"] = user.get("department") or ""
            return redirect(url_for("dashboard"))
        flash("Invalid login ID or password.", "error")
    return render_template("index.html", page="login")


@app.route("/dashboard")
@login_required
def dashboard():
    role = session["role"]
    user_id = session["user_id"]
    connection = connect_database()
    cursor = connection.cursor(dictionary=True)
    jobs, applications, users, complaints = [], [], [], []
    schedules, teacher_applications = [], []
    teachers, students, assignments = [], [], []

    try:
        if role == "student":
            cursor.execute("SELECT * FROM job_openings WHERE is_active = TRUE ORDER BY created_at DESC")
            jobs = cursor.fetchall()
            cursor.execute("""
                SELECT ja.*, jo.title AS job_title
                FROM job_applications ja
                JOIN job_openings jo ON ja.job_id = jo.job_id
                WHERE ja.student_id = %s
                ORDER BY ja.applied_at DESC
            """, (user_id,))
            applications = cursor.fetchall()
            cursor.execute("SELECT * FROM complaints WHERE student_id = %s ORDER BY created_at DESC", (user_id,))
            complaints = cursor.fetchall()
            cursor.execute("""
                SELECT cs.class_day, cs.start_time, cs.end_time, cs.classroom,
                       ta.subject, u.full_name AS teacher_name,
                       u.login_id AS teacher_login_id
                FROM teacher_assignments ta
                JOIN class_schedules cs ON ta.assignment_id = cs.assignment_id
                JOIN users u ON ta.teacher_id = u.user_id
                WHERE ta.student_id = %s
                ORDER BY FIELD(cs.class_day,'Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'), cs.start_time
            """, (user_id,))
            schedules = cursor.fetchall()

        elif role == "teacher":
            cursor.execute("SELECT * FROM job_openings WHERE is_active = TRUE ORDER BY created_at DESC")
            jobs = cursor.fetchall()
            cursor.execute("""
                SELECT DISTINCT u.user_id, u.login_id, u.full_name, u.email, u.department
                FROM teacher_assignments ta
                JOIN users u ON ta.student_id = u.user_id
                WHERE ta.teacher_id = %s
                ORDER BY u.full_name
            """, (user_id,))
            users = cursor.fetchall()
            cursor.execute("""SELECT DISTINCT
                cs.class_day,
                cs.start_time,
                cs.end_time,
                cs.classroom,
                ta.subject
                FROM teacher_assignments ta
                JOIN class_schedules cs ON ta.assignment_id = cs.assignment_id
                WHERE ta.teacher_id = %s
                ORDER BY FIELD(cs.class_day,'Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'), cs.start_time
            """, (user_id,))
            schedules = cursor.fetchall()
            cursor.execute("""
                SELECT u.full_name, u.login_id, jo.title AS job_title, ja.applied_at
                FROM job_applications ja
                JOIN job_openings jo ON ja.job_id = jo.job_id
                JOIN users u ON ja.student_id = u.user_id
                ORDER BY ja.applied_at DESC
            """)
            teacher_applications = cursor.fetchall()

        elif role == "hr":
            cursor.execute("SELECT * FROM job_openings ORDER BY created_at DESC")
            jobs = cursor.fetchall()
            cursor.execute("""
                SELECT ja.*, jo.title AS job_title, u.full_name, u.login_id, u.department
                FROM job_applications ja
                JOIN job_openings jo ON ja.job_id = jo.job_id
                JOIN users u ON ja.student_id = u.user_id
                ORDER BY ja.applied_at DESC
            """)
            applications = cursor.fetchall()
            cursor.execute("""
                SELECT c.*, u.full_name, u.login_id
                FROM complaints c
                JOIN users u ON c.student_id = u.user_id
                ORDER BY c.created_at DESC
            """)
            complaints = cursor.fetchall()
            cursor.execute("SELECT user_id, login_id, full_name, department FROM users WHERE role = 'teacher' AND is_active = TRUE ORDER BY full_name")
            teachers = cursor.fetchall()
            cursor.execute("SELECT user_id, login_id, full_name, department FROM users WHERE role = 'student' AND is_active = TRUE ORDER BY full_name")
            students = cursor.fetchall()
            cursor.execute("""
                    SELECT ta.assignment_id, ta.subject,
                    t.full_name AS teacher_name, t.login_id AS teacher_login_id,
                    s.full_name AS student_name, s.login_id AS student_login_id,
                    cs.class_day, cs.start_time, cs.end_time, cs.classroom
                    FROM teacher_assignments ta
                    JOIN users t ON ta.teacher_id = t.user_id
                    JOIN users s ON ta.student_id = s.user_id
                    LEFT JOIN class_schedules cs ON ta.assignment_id = cs.assignment_id
                    ORDER BY t.full_name, s.full_name
                """)
            assignments = cursor.fetchall()
            assignments = cursor.fetchall()

        elif role == "admin":
            search_id = request.args.get("search_id", "").strip()
            if search_id:
                cursor.execute("SELECT * FROM users WHERE login_id = %s", (search_id,))
                users = cursor.fetchall()
            cursor.execute("SELECT * FROM job_openings ORDER BY created_at DESC")
            jobs = cursor.fetchall()
    finally:
        cursor.close()
        connection.close()

    return render_template(
        "index.html", page="dashboard", role=role, jobs=jobs,
        applications=applications, users=users, complaints=complaints,
        schedules=schedules, teacher_applications=teacher_applications,
        teachers=teachers, students=students, assignments=assignments
    )


@app.route("/add_job", methods=["POST"])
@login_required
@role_required("hr")
def add_job():
    title = request.form.get("title", "").strip()
    department = request.form.get("department", "").strip()
    description = request.form.get("description", "").strip()
    required_skills = request.form.get("required_skills", "").strip()
    if not all([title, description, required_skills]):
        flash("Please fill in all required job fields.", "error")
        return redirect(url_for("dashboard"))
    connection = connect_database()
    cursor = connection.cursor()
    try:
        cursor.execute("""
            INSERT INTO job_openings (title, department, description, required_skills, created_by)
            VALUES (%s, %s, %s, %s, %s)
        """, (title, department, description, required_skills, session["user_id"]))
        connection.commit()
        flash("Job opening created successfully.", "success")
    except Exception as error:
        connection.rollback()
        print("Add job error:", error)
        flash("Could not create the job opening.", "error")
    finally:
        cursor.close()
        connection.close()
    return redirect(url_for("dashboard"))

@app.route("/delete_job/<int:job_id>", methods=["POST"])
@login_required
@role_required("hr")
def delete_job(job_id):
    connection = connect_database()
    cursor = connection.cursor()
    try:
        cursor.execute("SELECT COUNT(*) FROM job_applications WHERE job_id = %s", (job_id,))
        application_count = cursor.fetchone()[0]
        if application_count > 0:
            cursor.execute("UPDATE job_openings SET is_active = 0 WHERE job_id = %s", (job_id,))
            flash("Job has existing applications, so it was closed instead of deleted.", "success")
        else:
            cursor.execute("DELETE FROM job_openings WHERE job_id = %s", (job_id,))
            flash("Job opening removed successfully.", "success")
        connection.commit()
    except Exception as error:
        connection.rollback()
        print("Delete job error:", error)
        flash("Could not remove this job.", "error")
    finally:
        cursor.close()
        connection.close()
    return redirect(url_for("dashboard"))

@app.route("/apply_job/<int:job_id>", methods=["POST"])
@login_required
@role_required("student")
def apply_job(job_id):
    file = request.files.get("resume")
    if not file or not file.filename:
        flash("Please select your PDF resume.", "error")
        return redirect(url_for("dashboard"))
    if not file.filename.lower().endswith(".pdf"):
        flash("Only PDF resumes are allowed.", "error")
        return redirect(url_for("dashboard"))
    original_filename = secure_filename(file.filename)
    if not original_filename:
        flash("Invalid filename.", "error")
        return redirect(url_for("dashboard"))
    file_bytes = file.read()
    if not file_bytes.startswith(b"%PDF-"):
        flash("The uploaded file is not a valid PDF.", "error")
        return redirect(url_for("dashboard"))
    if len(file_bytes) > app.config["MAX_CONTENT_LENGTH"]:
        flash("The resume must be smaller than 5 MB.", "error")
        return redirect(url_for("dashboard"))

    connection = connect_database()
    cursor = connection.cursor(dictionary=True)
    resume_path = None
    try:
        cursor.execute("SELECT * FROM job_openings WHERE job_id = %s AND is_active = TRUE", (job_id,))
        job = cursor.fetchone()
        if not job:
            flash("This job is no longer available.", "error")
            return redirect(url_for("dashboard"))
        cursor.execute("SELECT application_id FROM job_applications WHERE job_id = %s AND student_id = %s", (job_id, session["user_id"]))
        if cursor.fetchone():
            flash("You have already applied for this job.", "error")
            return redirect(url_for("dashboard"))

        stored_filename = f"{uuid.uuid4().hex}.pdf"
        resume_path = UPLOAD_FOLDER / stored_filename
        with open(resume_path, "wb") as pdf_file:
            pdf_file.write(file_bytes)
        resume_text = extract_pdf_text(resume_path)
        analysis = analyze_resume(resume_text, job["description"], job["required_skills"])
        cursor.execute("""
            INSERT INTO job_applications
            (job_id, student_id, original_filename, stored_filename, extracted_text,
             match_score, text_similarity, matched_skills, missing_skills)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            job_id, session["user_id"], original_filename, stored_filename,
            resume_text, analysis["match_score"], analysis["text_similarity"],
            ", ".join(analysis["matched_skills"]), ", ".join(analysis["missing_skills"])
        ))
        connection.commit()
        flash(f"Application submitted! Your resume match score is {analysis['match_score']}%.", "success")
    except Exception as error:
        connection.rollback()
        if resume_path and resume_path.exists():
            resume_path.unlink()
        print("Resume upload error:", error)
        flash("Could not process your resume. Please check your PDF and try again.", "error")
    finally:
        cursor.close()
        connection.close()
    return redirect(url_for("dashboard"))


@app.route("/update_application/<int:application_id>", methods=["POST"])
@login_required
@role_required("hr")
def update_application(application_id):
    status = request.form.get("status")
    if status not in ["Pending", "Shortlisted", "Rejected"]:
        flash("Invalid application status.", "error")
        return redirect(url_for("dashboard"))
    connection = connect_database()
    cursor = connection.cursor()
    try:
        cursor.execute("UPDATE job_applications SET status = %s WHERE application_id = %s", (status, application_id))
        connection.commit()
        flash("Application status updated.", "success")
    finally:
        cursor.close()
        connection.close()
    return redirect(url_for("dashboard"))


@app.route("/resume/<int:application_id>")
@login_required
def resume_file(application_id):
    if session.get("role") == "teacher":
        abort(403)
    connection = connect_database()
    cursor = connection.cursor(dictionary=True)
    try:
        cursor.execute("SELECT * FROM job_applications WHERE application_id = %s", (application_id,))
        application = cursor.fetchone()
    finally:
        cursor.close()
        connection.close()
    if not application:
        abort(404)
    allowed = session["role"] in ["hr", "admin"] or (
        session["role"] == "student" and application["student_id"] == session["user_id"]
    )
    if not allowed:
        abort(403)
    return send_from_directory(
        UPLOAD_FOLDER, application["stored_filename"],
        as_attachment=True, download_name=application["original_filename"]
    )


@app.route("/submit_complaint", methods=["POST"])
@login_required
@role_required("student")
def submit_complaint():
    subject = request.form.get("subject", "").strip()
    message = request.form.get("message", "").strip()
    if not subject or not message:
        flash("Please enter the complaint subject and message.", "error")
        return redirect(url_for("dashboard"))
    connection = connect_database()
    cursor = connection.cursor()
    try:
        cursor.execute("INSERT INTO complaints (student_id, subject, message) VALUES (%s, %s, %s)", (session["user_id"], subject, message))
        connection.commit()
        flash("Your complaint has been submitted.", "success")
    finally:
        cursor.close()
        connection.close()
    return redirect(url_for("dashboard"))


@app.route("/update_complaint/<int:complaint_id>", methods=["POST"])
@login_required
@role_required("hr")
def update_complaint(complaint_id):
    status = request.form.get("status")
    if status not in ["Pending", "In Progress", "Resolved"]:
        flash("Invalid complaint status.", "error")
        return redirect(url_for("dashboard"))
    connection = connect_database()
    cursor = connection.cursor()
    try:
        cursor.execute("UPDATE complaints SET status = %s WHERE complaint_id = %s", (status, complaint_id))
        connection.commit()
        flash("Complaint status updated.", "success")
    finally:
        cursor.close()
        connection.close()
    return redirect(url_for("dashboard"))




@app.route("/add_assignment", methods=["POST"])
@login_required
@role_required("hr")
def add_assignment():
    teacher_id = request.form.get("teacher_id", type=int)
    student_ids = request.form.getlist("student_ids")
    subject = request.form.get("subject", "").strip()
    class_day = request.form.get("class_day", "").strip()
    start_time = request.form.get("start_time", "").strip()
    end_time = request.form.get("end_time", "").strip()
    classroom = request.form.get("classroom", "").strip()

    if not all([teacher_id, student_ids, subject, class_day, start_time, end_time]):
        flash("Please complete all required fields.", "error")
        return redirect(url_for("dashboard"))
    if class_day not in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"] or start_time >= end_time:
        flash("Please enter a valid class day and timing.", "error")
        return redirect(url_for("dashboard"))

    try:
        student_ids = list(set(int(s) for s in student_ids))
    except ValueError:
        flash("Invalid student selection.", "error")
        return redirect(url_for("dashboard"))

    connection = connect_database()
    cursor = connection.cursor(dictionary=True)
    try:
        cursor.execute("SELECT user_id FROM users WHERE user_id = %s AND role = 'teacher' AND is_active = TRUE", (teacher_id,))
        if not cursor.fetchone():
            flash("Select a valid teacher.", "error")
            return redirect(url_for("dashboard"))

        placeholders = ",".join(["%s"] * len(student_ids))
        cursor.execute(f"SELECT user_id FROM users WHERE user_id IN ({placeholders}) AND role = 'student' AND is_active = TRUE", tuple(student_ids))
        valid_students = [row["user_id"] for row in cursor.fetchall()]

        if len(valid_students) != len(student_ids):
            flash("One or more selected students are invalid.", "error")
            return redirect(url_for("dashboard"))

        added = 0
        for student_id in valid_students:
            cursor.execute("""
                SELECT assignment_id FROM teacher_assignments
                WHERE teacher_id = %s AND student_id = %s AND subject = %s
            """, (teacher_id, student_id, subject))
            if cursor.fetchone():
                continue

            cursor.execute("""
                INSERT INTO teacher_assignments (teacher_id, student_id, subject)
                VALUES (%s, %s, %s)
            """, (teacher_id, student_id, subject))
            assignment_id = cursor.lastrowid

            cursor.execute("""
                INSERT INTO class_schedules (assignment_id, class_day, start_time, end_time, classroom)
                VALUES (%s, %s, %s, %s, %s)
            """, (assignment_id, class_day, start_time, end_time, classroom or None))
            added += 1

        connection.commit()
        flash(f"Timetable assigned to {added} student(s).", "success")
    except Exception as error:
        connection.rollback()
        print("Assignment error:", error)
        flash("Could not save the assignments.", "error")
    finally:
        cursor.close()
        connection.close()
    return redirect(url_for("dashboard"))

@app.route("/delete_assignment/<int:assignment_id>", methods=["POST"])
@login_required
@role_required("hr")
def delete_assignment(assignment_id):
    connection = connect_database()
    cursor = connection.cursor()
    try:
        cursor.execute("DELETE FROM teacher_assignments WHERE assignment_id = %s", (assignment_id,))
        connection.commit()
        flash("Teacher assignment and its timetable removed.", "success")
    except Exception as error:
        connection.rollback()
        print("Delete assignment error:", error)
        flash("Could not remove the assignment.", "error")
    finally:
        cursor.close()
        connection.close()
    return redirect(url_for("dashboard"))


@app.route("/add_user", methods=["POST"])
@login_required
@role_required("admin")
def add_user():
    login_id = request.form.get("login_id", "").strip()
    full_name = request.form.get("full_name", "").strip()
    email = request.form.get("email", "").strip()
    password = request.form.get("password", "")
    role = request.form.get("role", "")
    department = request.form.get("department", "").strip()
    if not all([login_id, full_name, password, role]):
        flash("Please fill in the required fields.", "error")
        return redirect(url_for("dashboard"))
    if role not in ["student", "teacher"]:
        flash("Admin can create student and teacher accounts only.", "error")
        return redirect(url_for("dashboard"))
    connection = connect_database()
    cursor = connection.cursor()
    try:
        cursor.execute("""
            INSERT INTO users (login_id, full_name, email, password_hash, role, department)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (login_id, full_name, email or None, password, role, department))
        connection.commit()
        flash("User account created successfully.", "success")
    except Exception as error:
        connection.rollback()
        print("Add user error:", error)
        flash("Could not create user. Check for duplicate login ID or email.", "error")
    finally:
        cursor.close()
        connection.close()
    return redirect(url_for("dashboard"))


@app.route("/edit_user/<int:user_id>", methods=["POST"])
@login_required
@role_required("admin")
def edit_user(user_id):
    full_name = request.form.get("full_name", "").strip()
    email = request.form.get("email", "").strip()
    department = request.form.get("department", "").strip()
    role = request.form.get("role", "").strip()

    if not full_name or role not in ["student", "teacher"]:
        flash("Enter valid user details.", "error")
        return redirect(url_for("dashboard", search_id=request.form.get("login_id", "")))

    connection = connect_database()
    cursor = connection.cursor()
    try:
        cursor.execute("""
            UPDATE users
            SET full_name = %s, email = %s, department = %s, role = %s
            WHERE user_id = %s AND role IN ('student','teacher')
        """, (full_name, email or None, department, role, user_id))
        connection.commit()
        flash("User details updated successfully.", "success")
    except Exception as error:
        connection.rollback()
        print("Edit user error:", error)
        flash("Could not update user details.", "error")
    finally:
        cursor.close()
        connection.close()

    return redirect(url_for("dashboard", search_id=request.form.get("login_id", "")))

@app.route("/delete_user/<int:user_id>", methods=["POST"])
@login_required
@role_required("admin")
def delete_user(user_id):
    connection = connect_database()
    cursor = connection.cursor(dictionary=True)
    try:
        cursor.execute("SELECT user_id, role FROM users WHERE user_id = %s", (user_id,))
        user = cursor.fetchone()

        if not user:
            flash("User not found.", "error")
        elif user["role"] not in ["student", "teacher"]:
            flash("Protected accounts cannot be deleted here.", "error")
        else:
            cursor.execute("DELETE FROM users WHERE user_id = %s", (user_id,))
            connection.commit()
            flash("User account deleted successfully.", "success")
    except Exception as error:
        connection.rollback()
        print("Delete user error:", error)
        flash("Could not delete this user. Check whether related records prevent deletion.", "error")
    finally:
        cursor.close()
        connection.close()

    return redirect(url_for("dashboard"))


@app.route("/change_status/<int:user_id>", methods=["POST"])
@login_required
@role_required("admin")
def change_status(user_id):
    connection = connect_database()
    cursor = connection.cursor(dictionary=True)
    try:
        cursor.execute("SELECT user_id, role, is_active FROM users WHERE user_id = %s", (user_id,))
        user = cursor.fetchone()
        if not user:
            flash("User not found.", "error")
            return redirect(url_for("dashboard"))
        if user["role"] in ["hr", "admin"]:
            flash("HR and Admin accounts cannot be changed here.", "error")
            return redirect(url_for("dashboard"))
        cursor.execute("UPDATE users SET is_active = %s WHERE user_id = %s", (not bool(user["is_active"]), user_id))
        connection.commit()
        flash("Account status updated.", "success")
    finally:
        cursor.close()
        connection.close()
    return redirect(url_for("dashboard"))

@app.route("/delete_application/<int:application_id>", methods=["POST"])
@login_required
@role_required("hr")
def delete_application(application_id):
    connection = connect_database()
    cursor = connection.cursor(dictionary=True)
    try:
        cursor.execute("SELECT stored_filename, status FROM job_applications WHERE application_id = %s", (application_id,))
        application = cursor.fetchone()
        if not application:
            flash("Application not found.", "error")
        elif application["status"] != "Rejected":
            flash("Only rejected applications can be removed.", "error")
        else:
            cursor.execute("DELETE FROM job_applications WHERE application_id = %s", (application_id,))
            connection.commit()
            file_path = UPLOAD_FOLDER / application["stored_filename"]
            if file_path.exists():
                file_path.unlink()
            flash("Rejected application removed successfully.", "success")
    except Exception as error:
        connection.rollback()
        print("Delete application error:", error)
        flash("Could not remove the rejected application.", "error")
    finally:
        cursor.close()
        connection.close()
    return redirect(url_for("dashboard"))


@app.route("/logout")
def logout():
    session.clear()
    flash("You have logged out successfully.", "success")
    return redirect(url_for("home"))


@app.errorhandler(403)
def forbidden(error):
    return "You do not have permission to access this page.", 403


@app.errorhandler(404)
def not_found(error):
    return "The requested page or file was not found.", 404


@app.errorhandler(413)
def file_too_large(error):
    return "File is too large. Maximum allowed size is 5 MB.", 413


if __name__ == "__main__":
    app.run(debug=True)
