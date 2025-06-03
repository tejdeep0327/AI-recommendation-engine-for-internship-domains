import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample internship domains and their required skills
data = {
    'domain': [
        'Web Development', 'Machine Learning', 'Artificial Intelligence', 'Cloud Computing',
        'UI/UX Design', 'Cyber Security', 'Data Science', 'Mobile App Development'
    ],
    'skills_required': [
        'HTML CSS JavaScript React',
        'Python Pandas NumPy Scikit-learn',
        'Python NLP Deep Learning Neural Networks',
        'AWS Azure Docker DevOps',
        'Figma AdobeXD Wireframing Prototyping',
        'Network Security Ethical Hacking Firewalls',
        'Python Statistics Data Visualization Machine Learning',
        'Java Kotlin Flutter Firebase'
    ]
}

df = pd.DataFrame(data)


user_input = input("Enter your skills and interests (comma or space separated):\n").lower()


df = df.append({'domain': 'User', 'skills_required': user_input}, ignore_index=True)


tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df['skills_required'])

cos_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
recommended_index = cos_sim[0].argsort()[::-1]


print("\nTop Internship Recommendations for You:")
for idx in recommended_index[:3]:
    print(f"- {df.iloc[idx]['domain']}")

