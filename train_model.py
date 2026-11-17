from pathlib import Path
import joblib
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent
MODEL_FOLDER = BASE_DIR / "train_model"
MODEL_FOLDER.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_FOLDER / "resume_vectorizer.joblib"

resume_model = HashingVectorizer(
    n_features=2**18,
    alternate_sign=False,
    norm="l2",
    ngram_range=(1, 2),
    stop_words="english"
)

joblib.dump(resume_model, MODEL_PATH)

print("RESUME ANALYSIS MODEL CREATED!")
print("Model location:", MODEL_PATH)

resume_text = "Python machine learning SQL data analysis"
job_text = "Python SQL machine learning and data science"

resume_vector = resume_model.transform([resume_text])
job_vector = resume_model.transform([job_text])

similarity = cosine_similarity(resume_vector, job_vector)[0][0]
print("Sample text similarity:", round(similarity * 100, 2), "%")
