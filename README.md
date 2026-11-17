# Office Management System

## Project Overview

The **Office Management System** is a role-based web application developed to simplify and manage office and educational institution activities through a single interface.

The system provides four different user roles: **Student, Teacher, HR, and Admin**. Each role has access to specific features and functionalities based on its responsibilities.

The project also integrates Machine Learning techniques to analyze student resumes and evaluate their suitability for available job opportunities.

## User Roles and Features

### 1. Student
- View personal profile and class timetable.
- View available job openings.
- Apply for jobs by uploading PDF resumes.
- View resume application status.
- Submit complaints.

### 2. Teacher
- View teacher profile and department details.
- View assigned students.
- Check class schedules, subjects, timings, and classrooms.
- View available job openings.
- View student job application details.

### 3. HR
- Create and manage job openings.
- View student job applications.
- Analyze uploaded resumes using Machine Learning.
- View matched skills and missing skills.
- View resume match scores and text similarity scores.
- Shortlist or reject candidates.
- Manage student complaints.
- Assign teachers and students to classes.
- Create class schedules and manage classroom details.

### 4. Admin
- Create student and teacher accounts.
- Search users using Login ID.
- Update user information.
- Delete student and teacher accounts.
- Manage user records.

## Machine Learning Integration

The system includes an AI-based Resume Analysis module to evaluate student resumes against job requirements.

The analysis includes:
- Required skill identification.
- Matched skills detection.
- Missing skills identification.
- Resume skill match score.
- Text similarity score.

The results help HR evaluate applications more efficiently.

## Technologies Used

- **Programming Language:** Python
- **Frontend:** HTML, CSS
- **Backend Framework:** Flask
- **Database:** MySQL
- **Machine Learning:** Scikit-learn
- **Resume Processing:** PyPDF
- **Model Storage:** Joblib
- **Development Environment:** Visual Studio Code

## Project Structure

```text
Office-Management-System/
│
├── app.py
├── database.py
├── train_model.py
├── requirements.txt
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
└── train_model
```
