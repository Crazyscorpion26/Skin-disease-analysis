from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length, EqualTo, ValidationError
from werkzeug.security import generate_password_hash, check_password_hash
from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, EqualTo, Length
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os
import uuid
import secrets

app = Flask(__name__)

# Set a secret key for session management
app.secret_key = secrets.token_hex(16)

# Ensure the 'static/uploads' folder exists
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Database setup
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

# User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

# Load the model
try:
    model = load_model('skin_cancer_model.h5')
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

# Preprocessing function
IMG_SIZE = 128

def preprocess_image(image_path):
    try:
        img = Image.open(image_path)
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img = np.array(img) / 255.0  # Normalize image
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        return img
    except Exception as e:
        raise ValueError(f"Error processing image: {e}")
    

# Signup form
class SignupForm(FlaskForm):
    username = StringField('Username', validators=[InputRequired(), Length(min=3, max=50)])
    password = PasswordField('Password', validators=[InputRequired(), Length(min=6)])
    confirm_password = PasswordField('Confirm Password', validators=[
        InputRequired(), EqualTo('password', message='Passwords must match')
    ])
    submit = SubmitField('Sign Up')

# Login form
class LoginForm(FlaskForm):
    username = StringField('Username', validators=[InputRequired()])
    password = PasswordField('Password', validators=[InputRequired()])
    submit = SubmitField('Log In')

# Routes
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    form = SignupForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        new_user = User(username=form.username.data, password=hashed_password)
        try:
            db.session.add(new_user)
            db.session.commit()
            flash('Account created successfully!', 'success')
            return redirect(url_for('login'))
        except:
            flash('Username already exists.', 'danger')
    return render_template('signup.html', form=form)

def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        
        # Console log user data
        print(f"Attempted login with username: {form.username.data}")
        if user:
            print(f"User found: {user.username}, {user.id}")
        else:
            print("No user found with this username.")
        
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            session['user_id'] = user.username
            flash('Logged in successfully!', 'success')
            return redirect(url_for('index'))  # Redirect to the index page
        flash('Invalid username or password.', 'danger')
    return render_template('login.html', form=form)


@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('Logged out successfully!', 'success')
    return redirect(url_for('login'))

@app.route('/')
def index():
    if 'user_id' in session:  # Check if user is logged in
        return render_template('index.html')  # Show the index page
    flash('Please log in to access this page.', 'warning')
    return redirect(url_for('login')) 


if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # Ensures tables are created

# Disease details dictionary
DISEASE_INFO = {
    'Erysipelas': {
        'description': 'Bacterial skin infection causing redness and swelling.',
        'treatment': 'Antibiotics prescribed by a healthcare provider.',
        'first_action': 'Consult a doctor for antibiotic treatment.'
    },
    'Melanoma': {
        'description': 'A serious type of skin cancer that can spread to other organs.',
        'treatment': 'Surgical removal, immunotherapy, radiation, or chemotherapy.',
        'first_action': 'Schedule an appointment with a dermatologist immediately.'
    },
    'Impetigo': {
        'description': 'A highly contagious skin infection that causes red sores on the face.',
        'treatment': 'Antibiotics shorten the infection and help prevent spread to others.',
        'first_action': 'Clean the affected area and consult a doctor.'
    },
    'Folliculitis': {
        'description': 'An infection of the hair follicles that causes red, inflamed bumps.',
        'treatment': 'Topical antibiotics or antifungal creams may be used.',
        'first_action': 'Keep the area clean and avoid shaving until healed.'
    },
    'Erysipelas': {
        'description': 'A bacterial skin infection characterized by red, swollen patches.',
        'treatment': 'Antibiotics are necessary to treat the infection.',
        'first_action': 'Consult a healthcare provider for diagnosis and treatment.'
    },
    'Boils (Furunculosis)': {
        'description': 'Painful, pus-filled bumps that form under the skin due to infection.',
        'treatment': 'Warm compresses can help, and antibiotics may be needed.',
        'first_action': 'Do not squeeze; consult a doctor if it worsens.'
    },
    'Cellulitis': {
        'description': 'A common bacterial skin infection causing redness and swelling.',
        'treatment': 'Oral or intravenous antibiotics are typically required.',
        'first_action': 'Seek medical attention if you suspect cellulitis.'
    },
    'Impetigo': {
        'description': 'A highly contagious skin infection that causes red sores, often around the nose and mouth.',
        'treatment': 'Antibiotics shorten the infection and help prevent spread to others.',
        'first_action': 'Clean the affected area and consult a doctor.'
    },
    'Necrotizing Fasciitis': {
        'description': 'A severe bacterial infection that destroys skin, fat, and tissue.',
        'treatment': 'Immediate surgical intervention and broad-spectrum antibiotics are critical.',
        'first_action': 'Seek emergency medical care immediately.'
    },
    'Ringworm (Tinea corporis)': {
        'description': 'A fungal infection that causes a ring-shaped, red, itchy rash.',
        'treatment': 'Antifungal creams or oral medications are effective.',
        'first_action': 'Keep the area dry and consult a doctor for treatment.'
    },
    'Athlete’s Foot (Tinea pedis)': {
        'description': 'A fungal infection that causes itching, burning, and cracked skin on the feet.',
        'treatment': 'Antifungal powders or creams are commonly used.',
        'first_action': 'Keep feet dry and consult a pharmacist or doctor.'
    },
    'Jock Itch (Tinea cruris)': {
        'description': 'A fungal infection causing an itchy rash in the groin area.',
        'treatment': 'Antifungal creams or powders are effective.',
        'first_action': 'Keep the area dry and consult a healthcare provider if needed.'
    },
    'Scalp Ringworm (Tinea capitis)': {
        'description': 'A fungal infection of the scalp that can cause hair loss.',
        'treatment': 'Oral antifungal medications are often required.',
        'first_action': 'Consult a doctor for diagnosis and treatment options.'
    },
    'Onychomycosis (Nail Fungus)': {
        'description': 'A fungal infection of the nails that causes discoloration and thickening.',
        'treatment': 'Oral antifungal medications or topical treatments may be used.',
        'first_action': 'Consult a dermatologist for appropriate treatment.'
    },
    'Candidiasis': {
        'description': 'A fungal infection caused by Candida, often affecting moist areas of the body.',
        'treatment': 'Antifungal medications are used to treat the infection.',
        'first_action': 'Consult a healthcare provider for diagnosis and treatment.'
    },
    'Herpes Simplex Virus (Cold sores, Genital Herpes)': {
        'description': 'A viral infection causing painful blisters, typically around the mouth or genitals.',
        'treatment': 'Antiviral medications can help manage outbreaks.',
        'first_action': 'Consult a doctor for diagnosis and management options.'
    },
    'Varicella-Zoster Virus (Chickenpox, Shingles)': {
        'description': 'A viral infection causing an itchy rash and blisters; shingles is a reactivation of the virus.',
        'treatment': 'Antiviral medications can reduce severity and duration.',
        'first_action': 'Consult a healthcare provider for treatment options.'
    },
    'Molluscum Contagiosum': {
        'description': 'A viral skin infection causing small, raised, pearl-like bumps.',
        'treatment': 'Treatment may involve removal of lesions or topical therapies.',
        'first_action': 'Consult a dermatologist for evaluation and treatment.'
    },
    'Warts (Human Papillomavirus)': {
        'description': 'Benign growths on the skin caused by HPV, often appearing on hands and feet.',
        'treatment': 'Over-the-counter treatments or cryotherapy may be used.',
        'first_action': 'Consult a healthcare provider for persistent warts.'
    },
    'Hand, Foot, and Mouth Disease': {
        'description': 'A viral infection causing sores in the mouth and a rash on the hands and feet.',
        'treatment': 'Treatment focuses on relieving symptoms; hydration is important.',
        'first_action': 'Consult a pediatrician if symptoms are severe.'
    },
    'Scabies': {
        'description': 'A contagious skin condition caused by mites, leading to intense itching.',
        'treatment': 'Prescription creams or lotions are used to kill the mites.',
        'first_action': 'Consult a healthcare provider for diagnosis and treatment.'
    },
    'Cutaneous Larva Migrans': {
        'description': 'A skin infection caused by hookworm larvae, leading to itchy, winding tracks on the skin.',
        'treatment': 'Antiparasitic medications are effective.',
        'first_action': 'Consult a doctor for appropriate treatment.'
    },
    'Lice (Pediculosis)': {
        'description': 'Infestation of the scalp or body by lice, causing itching and irritation.',
        'treatment': 'Over-the-counter or prescription lice treatments are available.',
        'first_action': 'Use a lice comb and consult a healthcare provider if needed.'
    },
    'Leishmaniasis': {
        'description': 'A parasitic disease transmitted by sandflies, causing skin sores or systemic illness.',
        'treatment': 'Antiparasitic medications are required for treatment.',
        'first_action': 'Consult a healthcare provider for diagnosis and treatment.'
    },
    'Atopic Dermatitis (Eczema)': {
        'description': 'A chronic skin condition causing itchy, inflamed skin.',
        'treatment': 'Moisturizers and topical corticosteroids are commonly used.',
        'first_action': 'Avoid triggers and consult a dermatologist for management.'
    },
    'Contact Dermatitis': {
        'description': 'An allergic reaction causing red, itchy skin after contact with an irritant.',
        'treatment': 'Avoiding the irritant and using topical steroids can help.',
        'first_action': 'Identify and avoid the trigger; consult a doctor if severe.'
    },
    'Seborrheic Dermatitis': {
        'description': 'A common skin condition causing scaly patches and red skin, often on the scalp.',
        'treatment': 'Medicated shampoos and topical treatments are effective.',
        'first_action': 'Consult a dermatologist for appropriate treatment.'
    },
    'Nummular Eczema': {
        'description': 'A type of eczema characterized by round, coin-shaped spots on the skin.',
        'treatment': 'Moisturizers and topical steroids are commonly used.',
        'first_action': 'Consult a dermatologist for management options.'
    },
    'Plaque Psoriasis': {
        'description': 'A chronic autoimmune condition causing red, scaly patches on the skin.',
        'treatment': 'Topical treatments, phototherapy, and systemic medications may be used.',
        'first_action': 'Consult a dermatologist for a tailored treatment plan.'
    },
    'Guttate Psoriasis': {
        'description': 'A type of psoriasis that appears as small, drop-shaped lesions.',
        'treatment': 'Topical treatments and phototherapy are often effective.',
        'first_action': 'Consult a dermatologist for appropriate management.'
    },
    'Inverse Psoriasis': {
        'description': 'A form of psoriasis that occurs in skin folds, causing smooth, red patches.',
        'treatment': 'Topical treatments and lifestyle changes can help manage symptoms.',
        'first_action': 'Consult a dermatologist for tailored treatment options.'
    },
    'Pustular Psoriasis': {
        'description': 'A rare form of psoriasis characterized by white pustules surrounded by red skin.',
        'treatment': 'Systemic medications and topical treatments are often required.',
        'first_action': 'Seek medical advice for appropriate management.'
    },
    'Erythematotelangiectatic Rosacea': {
        'description': 'A subtype of rosacea characterized by redness and visible blood vessels.',
        'treatment': 'Topical treatments and lifestyle modifications can help manage symptoms.',
        'first_action': 'Consult a dermatologist for tailored treatment options.'
    },
    'Papulopustular Rosacea': {
        'description': 'A subtype of rosacea that causes red bumps and pustules on the face.',
        'treatment': 'Topical and oral medications can help control symptoms.',
        'first_action': 'Consult a dermatologist for appropriate management.'
    },
    'Lichen Planus': {
        'description': 'An inflammatory condition causing purplish, itchy, flat-topped bumps.',
        'treatment': 'Topical corticosteroids and other medications may be used.',
        'first_action': 'Consult a dermatologist for appropriate management.'
    }

    # Add more diseases as needed
}

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        flash("No file uploaded", "error")
        return redirect(url_for('index'))

    file = request.files['file']

    if file.filename == '':
        flash("No file selected", "error")
        return redirect(url_for('index'))

    # Save the uploaded image to the 'uploads' folder
    filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]  # Unique filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        # Preprocess the uploaded image
        img = preprocess_image(filepath)

        # Make the prediction
        prediction = model.predict(img)

        # Determine the predicted class and confidence
        predicted_class_index = np.argmax(prediction)
        confidence = np.max(prediction)

        # Mapping of model output indices to disease names
        class_names = ['Folliculitis', 'Erysipelas', 'Boils (Furunculosis)', 'Cellulitis', 'Impetigo', 'Necrotizing Fasciitis',
                       'Ringworm (Tinea corporis)', 'Athlete’s Foot (Tinea pedis)', 'Jock Itch (Tinea cruris)', 
                       'Scalp Ringworm (Tinea capitis)', 'Onychomycosis (Nail Fungus)', 'Candidiasis', 'Herpes Simplex Virus (Cold sores, Genital Herpes)',
                       'Varicella-Zoster Virus (Chickenpox, Shingles)', 'Molluscum Contagiosum', 'Warts (Human Papillomavirus)', 'Hand, Foot, and Mouth Disease',
                       'Scabies', 'Cutaneous Larva Migrans', 'Lice (Pediculosis)', 'Leishmaniasis', 'Atopic Dermatitis (Eczema)', 'Contact Dermatitis', 'Seborrheic Dermatitis', 'Nummular Eczema',
                       'Plaque Psoriasis', 'Guttate Psoriasis', 'Inverse Psoriasis', 'Pustular Psoriasis', 'Erythematotelangiectatic Rosacea', 'Papulopustular Rosacea', 'Lichen Planus']

        if predicted_class_index >= len(class_names):
            flash("Prediction failed due to unexpected class index", "error")
            return redirect(url_for('index'))

        predicted_class = class_names[predicted_class_index]
        disease_info = DISEASE_INFO.get(predicted_class, {})

        return render_template('result.html', prediction=predicted_class, confidence=f"{confidence * 100:.2f}%",
            image_filename=file.filename,
            description=disease_info.get('description', 'No description available.'),
            treatment=disease_info.get('treatment', 'No treatment information available.'),
            first_action=disease_info.get('first_action', 'No first action available.'),
        )



    except Exception as e:
        flash(f"Error during prediction: {e}", "error")
        return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)


