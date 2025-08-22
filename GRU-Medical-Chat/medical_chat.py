import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GRU, Dense, Embedding, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pickle
import re
import requests
import os

# Streamlit page configuration
st.set_page_config(page_title="Medical Health GRU Chatbot", page_icon="💊", layout="wide")

class MedicalGRUChatbot:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.max_sequence_length = 150
        self.vocab_size = 15000
        self.medical_keywords = self.load_medical_keywords()
        self.symptoms_db = self.load_symptoms_database()
        self.drug_db = self.load_drug_database()

    def load_medical_keywords(self):
        return {
            'symptoms': ['headache', 'fever', 'cough', 'fatigue', 'nausea', 
                         'dizziness', 'pain', 'ache','shortness of breath', 'chest pain', 'abdominal pain', 'back pain', 'sore throat',
                         'runny nose', 'congestion', 'vomiting', 'diarrhea','constipation', 'rash',
                         'swelling', 'joint pain', 'muscle pain', 'insomnia', 'anxiety', 'depression'],
            'body_parts': ['head', 'chest', 'abdomen', 'back', 'arm', 'leg','throat', 'stomach',
                           'heart', 'lungs', 'kidney', 'liver', 'brain', 'eye', 'ear', 'nose','mouth'], 
            'conditions': ['diabetes', 'hypertension', 'asthma', 'pneumonia','covid-19', 'flu',  
                          'cold', 'migraine', 'arthritis', 'depression', 'anxiety','allergies'], 
            'specialties': ['cardiology', 'neurology', 'gastroenterology','pulmonology', 'orthopedics', 
                           'dermatology', 'psychiatry', 'emergency medicine', 'family medicine'] 
        }

    def load_symptoms_database(self):
        return {
            'fever + cough + fatigue': {'conditions': ['Common Cold', 'Flu', 'COVID-19', 'Pneumonia'], 
                                        'recommendations': 'Rest, hydration, monitor temperature. Consult doctor if symptoms worsen.', 
                                        'urgency': 'moderate'},
            'chest pain + shortness of breath': {'conditions': ['Heart Attack', 'Angina', 'Pulmonary Embolism', 'Asthma'], 
                                                'recommendations': 'SEEK IMMEDIATE MEDICAL ATTENTION', 
                                                'urgency': 'high'},
            'headache + fever + stiff neck': {'conditions': ['Meningitis', 'Encephalitis'], 
                                             'recommendations': 'EMERGENCY - Go to hospital immediately', 
                                             'urgency': 'critical'},
            'abdominal pain + nausea + vomiting': {'conditions': ['Gastroenteritis', 'Appendicitis', 'Food Poisoning'], 
                                                  'recommendations': 'Monitor symptoms, stay hydrated. See doctor if severe or persistent.', 
                                                  'urgency': 'moderate'},
            'persistent cough + weight loss + night sweats': {'conditions': ['Tuberculosis', 'Lung Cancer', 'Chronic Infection'], 
                                                             'recommendations': 'Consult pulmonologist for thorough evaluation', 
                                                             'urgency': 'high'}
        }

    def load_drug_database(self):
        return {
            'paracetamol': {'uses': 'Pain relief, fever reduction', 'dosage': '500-1000mg every 4-6 hours, max 4g/day',
                            'side_effects': 'Liver damage with overdose', 'contraindications': 'Severe liver disease'},
            'ibuprofen': {'uses': 'Pain relief, anti-inflammatory, fever reduction', 'dosage': '200-400mg every 4-6 hours, max 1.2g/day',
                          'side_effects': 'Stomach upset, kidney problems', 'contraindications': 'Kidney disease, stomach ulcers'},
            'aspirin': {'uses': 'Pain relief, blood thinner, heart protection', 'dosage': '75-325mg daily for prevention, 500-1000mg for pain',
                        'side_effects': 'Bleeding risk, stomach irritation', 'contraindications': 'Bleeding disorders, children with viral infections'},
            'metformin': {'uses': 'Type 2 diabetes management', 'dosage': '500-850mg twice daily with meals',
                          'side_effects': 'Gastrointestinal upset, lactic acidosis (rare)', 'contraindications': 'Kidney disease, liver disease'},
            'lisinopril': {'uses': 'High blood pressure, heart failure', 'dosage': '2.5-40mg once daily',
                           'side_effects': 'Dry cough, hyperkalemia, angioedema', 'contraindications': 'Pregnancy, bilateral renal artery stenosis'}
        }

    def preprocess_medical_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s\-/]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def create_medical_model(self, vocab_size, embedding_dim=150, gru_units=300):
        model = Sequential([
            Embedding(vocab_size, embedding_dim, input_length=self.max_sequence_length),
            GRU(gru_units, return_sequences=True, dropout=0.3, recurrent_dropout=0.2),
            GRU(gru_units//2, return_sequences=True, dropout=0.3, recurrent_dropout=0.2),
            GRU(gru_units//4, dropout=0.3),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(vocab_size, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def prepare_medical_training_data(self, conversations):
        input_texts, target_texts = [], []
        for conversation in conversations:
            if len(conversation) >= 2:
                for i in range(len(conversation)-1):
                    input_texts.append(self.preprocess_medical_text(conversation[i]))
                    target_texts.append(self.preprocess_medical_text(conversation[i+1]))
        return input_texts, target_texts

    def train_model(self, input_texts, target_texts, model_file="medical_gru_model.h5", tokenizer_file="tokenizer.pkl"):
        """Train or load pre-trained model to save time"""
        if os.path.exists(model_file) and os.path.exists(tokenizer_file):
            self.model = load_model(model_file)
            with open(tokenizer_file, "rb") as f:
                self.tokenizer = pickle.load(f)
            return

        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token="<OOV>")
        all_texts = input_texts + target_texts
        self.tokenizer.fit_on_texts(all_texts)

        input_sequences = self.tokenizer.texts_to_sequences(input_texts)
        target_sequences = self.tokenizer.texts_to_sequences(target_texts)

        X = pad_sequences(input_sequences, maxlen=self.max_sequence_length, padding='post')
        y = pad_sequences(target_sequences, maxlen=self.max_sequence_length, padding='post')
        y = y[:, 0] if y.shape[1] > 0 else np.zeros(len(y))

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model = self.create_medical_model(len(self.tokenizer.word_index)+1)
        self.model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32, verbose=0)

        # Save model and tokenizer
        self.model.save(model_file)
        with open(tokenizer_file, "wb") as f:
            pickle.dump(self.tokenizer, f)

    # Use the updated response generator with limited GRU words
    def generate_medical_response(self, input_text, max_length=15):
        if not self.model or not self.tokenizer:
            return "Medical system not initialized. Please train the model first."

        input_lower = input_text.lower()

        # Emergencies
        emergency_keywords = ['chest pain', 'difficulty breathing', 'severe pain', 'unconscious',
                             'bleeding', 'poisoning', 'heart attack', 'stroke']
        for emergency in emergency_keywords:
            if emergency in input_lower:
                return ("EMERGENCY: Please call emergency services (911) or go to the nearest hospital immediately. "
                        "This chatbot cannot replace emergency medical care.")

        # Symptoms
        symptom_analysis = self.analyze_symptoms(input_text)
        if 'possible_conditions' in symptom_analysis:
            response = f"Based on your symptoms ({', '.join(symptom_analysis['symptoms'])}), possible conditions include: {', '.join(symptom_analysis['possible_conditions'])}.\n\n"
            response += f"Recommendation: {symptom_analysis['recommendations']}\n\n"
            response += "Please consult with a healthcare professional for proper diagnosis and treatment."
            return response

        # Drugs
        drug_keywords = ['medicine', 'medication', 'drug', 'pill', 'tablet']
        if any(keyword in input_lower for keyword in drug_keywords):
            for drug in self.drug_db.keys():
                if drug in input_lower:
                    drug_info = self.get_drug_info(drug)
                    if drug_info:
                        return (
                            f"**{drug.title()} Information:**\n\n"
                            f"Uses: {drug_info['uses']}\n"
                            f"Dosage: {drug_info['dosage']}\n"
                            f"Side Effects: {drug_info['side_effects']}\n"
                            f"Contraindications: {drug_info['contraindications']}\n\n"
                            "Always consult your doctor before taking any medication."
                        )

        # Short GRU response (optional)
        processed_input = self.preprocess_medical_text(input_text)
        sequence = self.tokenizer.texts_to_sequences([processed_input])
        padded_sequence = pad_sequences(sequence, maxlen=self.max_sequence_length, padding='post')

        response_words = []
        current_sequence = padded_sequence[0].tolist()

        for _ in range(max_length):
            prediction = self.model.predict(np.array([current_sequence]), verbose=0)[0]
            prediction = prediction / prediction.sum()
            next_word_idx = np.random.choice(len(prediction), p=prediction)
            next_word = None
            for word, idx in self.tokenizer.word_index.items():
                if idx == next_word_idx:
                    next_word = word
                    break
            if next_word and next_word != "<OOV>":
                response_words.append(next_word)
                current_sequence = current_sequence[1:] + [next_word_idx]
            else:
                break

        model_response = " ".join(response_words)
        if not model_response:
            model_response = "I understand your concern. Could you provide more details about your symptoms?"
        model_response += "\n\nMedical Disclaimer: This chatbot provides general health information only. Always consult qualified healthcare professionals for medical advice, diagnosis, and treatment."
        return model_response

    def analyze_symptoms(self, user_input):
        symptoms_found = []
        input_lower = user_input.lower()
        for symptom in self.medical_keywords['symptoms']:
            if symptom in input_lower:
                symptoms_found.append(symptom)

        for symptom_combo, info in self.symptoms_db.items():
            combo_symptoms = symptom_combo.split(' + ')
            if all(symptom in input_lower for symptom in combo_symptoms):
                return {
                    'symptoms': combo_symptoms,
                    'possible_conditions': info['conditions'],
                    'recommendations': info['recommendations'],
                    'urgency': info['urgency']
                }

        return {'symptoms': symptoms_found, 'analysis': 'partial'}

    def get_drug_info(self, drug_name):
        drug_lower = drug_name.lower()
        for drug, info in self.drug_db.items():
            if drug in drug_lower:
                return info
        return None


def get_health_data():
    data = {}
    try:
        health_tips = [
            "Drink at least 8 glasses of water daily",
            "Get 7-9 hours of sleep each night",
            "Exercise for at least 30 minutes daily",
            "Eat 5 servings of fruits and vegetables daily",
            "Practice stress management techniques",
            "Get regular health check-ups",
            "Maintain good hand hygiene",
            "Limit processed foods and sugar intake"
        ]
        data['daily_health_tip'] = np.random.choice(health_tips)
    except Exception as e:
        st.error(f"Error fetching health data: {e}")
    return data


def create_medical_conversations():
    conversations = [
        ["I have a headache", "I understand you're experiencing a headache. Can you describe the pain - is it throbbing, sharp, or dull? How long have you had it?"],
        ["I feel nauseous", "Nausea can be uncomfortable. Are you experiencing other symptoms like vomiting, fever, or abdominal pain?"],
        ["I have chest pain", "Chest pain requires immediate attention. Please describe the pain and consider seeking emergency medical care if it's severe."],
        ["I can't sleep", "Insomnia can affect your health. Have you tried establishing a regular sleep routine? Consider speaking with a healthcare provider."],
        ["I have a fever", "A fever indicates your body is fighting something. Monitor your temperature and stay hydrated. Seek medical care if it persists or is very high."],
        ["I feel dizzy", "Dizziness can have many causes. Are you experiencing other symptoms like headache, nausea, or blurred vision?"],
        ["I feel shortness of breath", "Shortness of breath may be serious. Consider seeking medical attention immediately."],
        ["My throat hurts", "A sore throat can be caused by infection or irritation. Gargling warm salt water may help, but see a doctor if pain persists."],
        ["I have a cough", "Coughing is common. Is it dry or producing mucus? Do you have fever or shortness of breath?"],
        ["I feel tired all the time", "Fatigue can be caused by many conditions. How long have you been feeling this way? Are you sleeping well?"],
        ["I feel anxious", "Anxiety is common and treatable. Consider relaxation techniques, exercise, and speaking with a mental health professional."],
        ["I feel depressed", "Depression is a serious condition. Please consider reaching out to a mental health professional or your primary care doctor."],
        ["I think I sprained my ankle", "For a sprain, rest, ice, compression, and elevation can help. Seek medical care if pain is severe or swelling persists."],
        ["I have abdominal pain", "Abdominal pain can have many causes. Is it sudden, sharp, or persistent? Any nausea or vomiting?"],
        ["I have diarrhea", "Diarrhea can lead to dehydration. Drink fluids and monitor symptoms. Consult a doctor if severe or persistent."],
        ["I feel bloated", "Bloating may be related to diet or digestive issues. Have you noticed any specific triggers?"],
        ["My joints hurt", "Joint pain can be due to injury, arthritis, or inflammation. How long have you been experiencing this pain?"],
        ["I have back pain", "Back pain is common. Can you describe the location and intensity? Any numbness or weakness?"],
        ["I have high blood pressure", "Managing hypertension is important. Are you taking medication, and do you monitor your blood pressure regularly?"],
        ["I have diabetes", "Diabetes management includes monitoring blood sugar, diet, and medication adherence. How is your glucose control?"],
        ["Can I take ibuprofen?", "Ibuprofen is used for pain relief and inflammation. Take with food and follow dosage instructions."],
        ["What is paracetamol used for?", "Paracetamol is used for pain relief and reducing fever. Follow recommended dosage instructions."],
        ["I have a rash", "Rashes can be caused by allergies or infections. Is it itchy, red, or spreading? Seek medical care if severe."],
        ["I feel muscle pain", "Muscle pain can be due to exercise, injury, or illness. How long have you experienced this pain?"],
        ["I have a fever and cough", "Based on your symptoms, possible conditions include: Common Cold, Flu, COVID-19. Rest, hydrate, and monitor your temperature."],
        ["I have nausea and vomiting", "Stay hydrated and avoid solid foods until symptoms improve. Seek medical attention if persistent."],
        ["I feel weakness", "Weakness can result from many conditions. How long have you felt weak, and do you have other symptoms?"],
        ["I have blurry vision", "Blurry vision may be due to eye strain or underlying conditions. Schedule an eye exam if persistent."],
        ["I feel dizzy when standing up", "This may be due to low blood pressure or dehydration. Stand slowly and drink fluids."],
        ["I have chest tightness", "Chest tightness may be serious. Seek medical attention immediately if severe or persistent."],
        ["I have insomnia", "Establish a regular sleep routine, limit caffeine, and avoid screens before bedtime. See a doctor if it continues."],
        ["I have allergies", "Allergy management may include avoiding triggers and taking antihistamines. Consult a doctor for severe reactions."],
        ["I feel heart palpitations", "Palpitations may be caused by stress or heart conditions. Seek medical advice if frequent or severe."],
        ["I have a sore throat and fever", "This may indicate an infection. Stay hydrated, rest, and see a doctor if symptoms worsen."],
        ["I feel dizzy and nauseous", "These symptoms may indicate dehydration, infection, or other conditions. Monitor your condition and consult a doctor if persistent."],
        ["I have ear pain", "Ear pain can be due to infection or pressure. See a doctor if severe or persistent."],
        ["I have stomach cramps", "Stomach cramps may be digestive-related. Monitor symptoms and consult a doctor if severe or persistent."],
        ["I feel cold and shivery", "Shivering and chills may indicate infection. Keep warm, hydrate, and monitor your temperature."],
        ["I have swelling in my leg", "Leg swelling can indicate injury or circulation issues. Elevate the leg and seek medical advice."],
        ["I feel tingling in my hands", "Tingling can be due to nerve issues or circulation problems. Consult a doctor if persistent."],
        ["I feel anxious and stressed", "Stress management techniques such as deep breathing, exercise, or counseling can help."],
        ["I have a headache and nausea", "Monitor symptoms, rest, and stay hydrated. Seek medical care if severe or persistent."],
        ["I have joint pain and swelling", "This may indicate inflammation or injury. Rest, ice, and consult a doctor if necessary."],
        ["I feel fatigue and weakness", "Evaluate your sleep, diet, and activity level. Persistent fatigue should be discussed with a doctor."],
        ["I have chest pain and shortness of breath", "This is serious. Seek immediate medical attention."],
        ["I feel dizzy and faint", "Sit or lie down immediately and seek medical advice. Could indicate low blood pressure or other conditions."],
        ["I have abdominal pain and diarrhea", "Stay hydrated, avoid irritants, and consult a doctor if severe or persistent."],
        ["I feel nauseous and have a fever", "Monitor your temperature, rest, and seek medical advice if persistent."],
        ["I have back pain and stiffness", "Gentle stretching, rest, and monitoring symptoms is recommended. Seek medical advice if persistent."],
        ["I feel heartburn", "Avoid spicy and acidic foods, eat smaller meals, and monitor symptoms."],
        ["I have difficulty breathing", "This is an emergency. Call emergency services immediately."]
    ]
    
    return conversations



def main():
    st.title("Medical Health GRU Chatbot")

    # Display daily health tip
    health_data = get_health_data()
    if 'daily_health_tip' in health_data:
        st.info(f"💡 Daily Health Tip: {health_data['daily_health_tip']}")

    # Initialize chatbot
    chatbot = MedicalGRUChatbot()

    # Train or load model
    st.info("⚙️ Loading or training medical GRU model. This may take a few seconds if first time...")
    conversations = create_medical_conversations()
    input_texts, target_texts = chatbot.prepare_medical_training_data(conversations)
    chatbot.train_model(input_texts, target_texts)
    st.success("✅ Medical GRU chatbot ready! Ask about symptoms or drugs.")

    # Chat input
    user_input = st.text_input("Ask me about symptoms or drugs:")

    if user_input:
        st.write(f"👨‍⚕️ You asked: {user_input}")
        response = chatbot.generate_medical_response(user_input)
        st.write(f"🤖 Bot: {response}")


if __name__ == "__main__":
    main()
