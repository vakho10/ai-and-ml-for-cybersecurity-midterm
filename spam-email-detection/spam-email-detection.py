import re

import joblib
import matplotlib.pyplot as plt
import nltk
import pandas as pd
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

# Download NLTK resources (only need to do this once)
nltk.download('stopwords')


# Function to load data from the local CSV file
def load_data(filename="v_laluashvili_791204.csv"):
    try:
        data = pd.read_csv(filename)
        print("Data loaded successfully!")
        return data
    except FileNotFoundError:
        print(f"File {filename} not found. Please make sure the file is in the same directory as the script.")
        return None


# Function to explore and visualize the data
def explore_data(data):
    print("\nData Head:")
    print(data.head())
    print("\nData Info:")
    print(data.info())
    print("\nData Description:")
    print(data.describe())

    # Check for missing values
    print("\nMissing Values:")
    print(data.isnull().sum())

    # Check the distribution of classes
    print("\nClass Distribution:")
    print(data['is_spam'].value_counts())

    # Visualization 1: Distribution of classes
    plt.figure(figsize=(8, 6))
    sns.countplot(x='is_spam', data=data)
    plt.title('Distribution of Email Classes')
    plt.xlabel('Class (0: Legitimate, 1: Spam)')
    plt.ylabel('Count')
    plt.savefig('class_distribution.png')
    plt.close()

    # Visualization 2: Feature correlation heatmap
    plt.figure(figsize=(12, 10))
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Feature Correlation Heatmap')
    plt.savefig('feature_correlation.png')
    plt.close()

    # Visualization 3: Box plots for each feature by class
    plt.figure(figsize=(15, 10))
    for i, column in enumerate(['words', 'links', 'capital_words', 'spam_word_count']):
        plt.subplot(2, 2, i + 1)
        sns.boxplot(x='is_spam', y=column, data=data)
        plt.title(f'{column} by Class')
    plt.tight_layout()
    plt.savefig('feature_boxplots.png')
    plt.close()


# Function to train a logistic regression model
def train_model(data):
    # Split the data into features and target
    X = data[['words', 'links', 'capital_words', 'spam_word_count']]
    y = data['is_spam']

    # Split the data into training and testing sets (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Create and train the logistic regression model
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    # Print the model coefficients
    print("\nModel Coefficients:")
    for feature, coef in zip(X.columns, model.coef_[0]):
        print(f"{feature}: {coef}")

    # Test the model on the test set
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy}")

    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # Visualize the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Legitimate', 'Spam'],
                yticklabels=['Legitimate', 'Spam'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()

    # Visualization: Feature importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_[0]
    })
    feature_importance = feature_importance.sort_values('Coefficient', ascending=False)

    plt.figure(figsize=(10, 8))
    sns.barplot(x='Coefficient', y='Feature', data=feature_importance)
    plt.title('Feature Importance (Logistic Regression Coefficients)')
    plt.savefig('feature_importance.png')
    plt.close()

    # Save the model for future use
    joblib.dump(model, 'spam_detection_model.pkl')

    return model, X.columns


# Function to preprocess email text
def preprocess_email(email_text):
    # Convert to lowercase
    email_text = email_text.lower()

    # Remove special characters, numbers, and punctuation
    email_text = re.sub(r'[^a-zA-Z\s]', '', email_text)

    # Tokenize the text
    words = email_text.split()

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # Stemming
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]

    return ' '.join(words)


# Function to extract features from email text
def extract_features(email_text):
    # Preprocess the email text
    processed_text = preprocess_email(email_text)

    # Create a dictionary to store feature values
    features = {}

    # Count total words
    features['words'] = len(email_text.split())

    # Count links (URLs)
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\,]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    features['links'] = len(re.findall(url_pattern, email_text))

    # Count capital words
    features['capital_words'] = sum(1 for word in email_text.split() if word.isupper())

    # Count spam words
    spam_words = ['free', 'win', 'prize', 'click', 'money', 'urgent', 'offer', 'discount', 'congratulations', 'claim']
    features['spam_word_count'] = sum(1 for word in processed_text.split() if word in spam_words)

    return features


# Function to classify an email
def classify_email(email_text, model):
    features = extract_features(email_text)
    feature_values = [features['words'], features['links'], features['capital_words'], features['spam_word_count']]
    prediction = model.predict([feature_values])[0]
    probability = model.predict_proba([feature_values])[0]

    return prediction, probability, features


# Function to test example emails
def test_example_emails(model):
    # Example spam email
    spam_email = """
    Congratulations! You've WON a $1,000,000 PRIZE! 
    CLICK here to claim your REWARD now! 
    This OFFER expires in 24 HOURS. 
    Don't miss this OPPORTUNITY of a LIFETIME!
    FREE MONEY for you!
    """

    # Example legitimate email
    legitimate_email = """
    Dear Team,

    I hope this email finds you well. I'm writing to inform you about our upcoming meeting scheduled for next Tuesday at 2 PM. 
    We will be discussing the Q3 financial results and planning for the next quarter. 
    Please prepare your reports and be ready to present your department's achievements.

    Best regards,
    John Smith
    """

    # Classify the example emails
    spam_prediction, spam_probability, spam_features = classify_email(spam_email, model)
    legitimate_prediction, legitimate_probability, legitimate_features = classify_email(legitimate_email, model)

    print(f"\nSpam Email Features: {spam_features}")
    print(f"Spam Email Prediction: {spam_prediction} with probability {spam_probability}")

    print(f"\nLegitimate Email Features: {legitimate_features}")
    print(f"Legitimate Email Prediction: {legitimate_prediction} with probability {legitimate_probability}")

    return spam_email, legitimate_email


# Main function
def main():
    # Load the data
    data = load_data()
    if data is None:
        return

    # Explore the data
    explore_data(data)

    # Train the model
    model, feature_columns = train_model(data)

    # Test example emails
    spam_email, legitimate_email = test_example_emails(model)

    # Interactive email classification
    print("\n=== Email Spam Detector ===")
    print("Enter an email text to classify, or 'quit' to exit:")

    while True:
        user_input = input("\nEnter email text: ")
        if user_input.lower() == 'quit':
            break

        prediction, probability, features = classify_email(user_input, model)
        print(f"Extracted Features: {features}")
        print(f"Prediction: {'Spam' if prediction == 1 else 'Legitimate'}")
        print(f"Probability: Spam - {probability[1]:.2f}, Legitimate - {probability[0]:.2f}")


if __name__ == "__main__":
    main()
