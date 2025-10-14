# Predictive Pokemon Battle Engine

**Date of Creation:** 14/10/2025

**Link to Repository:** https://github.com/TomADavis/Predictive-Pokemon-Battle-Engine

---

### ✨ About the Project

I have been really excited to start this project but also quite nervous. On job applications when it asks "Have you had any experience with data science concepts" I could never say 100% yes because I was also too intimidated by the idea of a machine learning model, without having tried to code one in the past I never knew how simple they could be and this is where this project started...

I built a machine learning model that predicts the winner of a one-on-one Pokémon battle, which was a really fun way to apply some serious concepts. The core of the project involved taking the base stats for the Generation 1 Pokémon, simulating over 27,000 unique battles, and then training a model to spot the patterns that lead to a win. Going through the whole process, from cleaning the data to evaluating the final model, was an amazing learning experience. It really showed me just how powerful libraries like Pandas and Scikit-learn are.

Now that I've built this predictor, I'm really excited to go back to my old A-Level Pokémon game, 'Fabled Realm', and see how I can rebuild its battle system with a much smarter, data-driven engine.

---

### ⚙️ Features

* **Data Cleaning:** The initial dataset is cleaned by handling missing values and filtering for a specific subset (Generation 1).
* **Feature Engineering:** Simulates over 27,000 unique battle matchups by creating a Cartesian product of the Pokémon dataset and engineering 'difference' features for model training.
* **Predictive Modelling:** Implements a Logistic Regression model from `scikit-learn` to classify the winner based on the differences in stats.
* **Model Evaluation:** The model's performance is rigorously tested using metrics like accuracy, a classification report, and a confusion matrix to visualize its predictive power.

---

### 📚 Libraries and Language Used

* **Language:** Python

* `pandas`: Used for all data loading, cleaning, manipulation, and the core feature engineering process.
* `matplotlib` & `seaborn`: Used together for data visualization, specifically for generating the confusion matrix heatmap to evaluate the model's performance.
* `scikit-learn`: The core machine learning library used for:
    * `train_test_split`: To separate the data into training and testing sets.
    * `LogisticRegression`: To build the classification model.
    * `accuracy_score`, `classification_report`, `confusion_matrix`: To evaluate the model's predictions.

---

### 🧠 What I Learned

* **Pandas Best Practices:** A major takeaway was the importance of writing safe and explicit Pandas code. I learned the critical difference between a **View and a Copy**, and why using `.copy()` is essential to prevent unintended changes to original DataFrames. I also learned the modern, safer way to handle operations like `fillna` by assigning the result back to the column instead of using `inplace=True`.

* **The Power of Feature Engineering:** The most creative part of the project was transforming a simple list of Pokémon stats into a rich dataset of battle matchups. I discovered clever techniques like using a temporary `'key'` column to create a Cartesian product, which was a great problem-solving insight.

* **Effective Data Visualization:** I gained a much deeper appreciation for the relationship between Seaborn and Matplotlib. I learned how high-level parameters like `hue` in Seaborn can simplify complex plotting tasks that would require manual loops and data filtering in pure Matplotlib.

* **Core Machine Learning Principles:** This project solidified my understanding of the machine learning workflow. I now understand *why* `train_test_split` is non-negotiable for honest model evaluation. Furthermore, learning to interpret a **confusion matrix** was a real eye-opener, as it provides so much more detail about a model's performance than a simple accuracy score.

---

### 🛠️ Prerequisites

1.  **Python 3:** To check if you have it, open your terminal or Command Prompt and type:
    ```bash
    python --version
    ```
2.  **Required Libraries:** To download the necessary libraries, open your terminal and type:
    ```bash
    pip install pandas matplotlib seaborn scikit-learn
    ```
3.  **The Dataset:** The project requires the `Pokemon.csv` file to be in the same directory as the script.

---

### ▶️ Running the Program

1.  Ensure you have met all the prerequisites listed above.
2.  Save the code as a Python file (e.g., `pokemon_predictor.py`).
3.  Place the `Pokemon.csv` file in the same folder.
4.  Navigate to the project folder in your terminal or Command Prompt.
5.  Run the main script:
    ```bash
    python pokemon_predictor.py
    ```
    The script will print the model's accuracy and classification report to the console and then display a plot of the confusion matrix.
