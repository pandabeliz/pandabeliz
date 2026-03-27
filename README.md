# Hi, I'm Beliz 👋

## 🤖 NLP & Transformer Specialist | Data Scientist

I'm a **Master's student in Applied Data Science** at Utrecht University, specializing in **Natural Language Processing, transformer architectures, and deep learning**. I build production-ready NLP systems using state-of-the-art models and apply them to real-world challenges across multiple domains.

🎓 **Background:** Cognitive Science & AI (BSc) → Applied Data Science (MSc)  
🌍 **Location:** Utrecht, Netherlands  
💡 **Expertise:** Fine-tuning transformers, multi-label text classification, embeddings, and semantic analysis  
🔬 **Research:** 18 experiments, 250 participants in computational linguistics study

---

## 🚀 Currently Working On

- 🤖 **Deep Diving into Transformers:** Exploring post-transformer architectures and attention mechanisms
- 📊 **Building NLP Portfolio:** Comprehensive showcase of transformer fine-tuning, text classification, and embeddings
- 📚 **Master's Thesis:** [Transformer-based text analysis project - topic in development]
- 🧠 **Learning:** Advanced transformer architectures (BERT variants, GPT, T5), prompt engineering, and LLM applications

---

## 🛠️ Technical Skills

### 🤖 **NLP & Transformers Expertise**

#### **Expert Level** ⭐⭐⭐
*Production-ready, proven in projects*

**Multi-Label Text Classification**
- Handled 54-category classification with 0.78 F1-score
- Implemented stratified sampling for class imbalance
- Custom loss weighting and threshold optimization
- Statistical significance testing across categories

**Transformer Fine-Tuning**
- DistilBERT for classification (custom heads)
- Sentence-Transformers for embeddings
- Hyperparameter optimization (grid search, LR scheduling)
- Attention mechanism implementation

**Text Representations**
- Comparative analysis: TF-IDF vs. Word2Vec vs. FastText vs. Transformers
- 300-dimensional word embeddings (Word2Vec, FastText)
- Transformer-based embeddings (768-dim from BERT-family)
- N-gram feature engineering (1-3 grams)

#### **Proficient** ⭐⭐
*Strong theoretical knowledge, some practical experience*

**Model Architectures**
- BERT family (BERT, DistilBERT, RoBERTa, ALBERT)
- Encoder-only transformers for classification
- Sentence-Transformers architecture
- Classical ML for text (Naive Bayes, SVM)

**NLP Fundamentals**
- Tokenization strategies (WordPiece, BPE, SentencePiece)
- Text preprocessing pipelines
- Feature extraction and selection
- Cross-validation and evaluation metrics

**Tools & Frameworks**
- Hugging Face Transformers library
- PyTorch for model training
- scikit-learn for classical ML
- pandas/NumPy for data manipulation

#### **Learning & Developing** ⭐
*Currently studying, can implement with guidance*

**Advanced Transformers**
- Decoder models (GPT-family)
- Encoder-decoder (T5, BART)
- Prompt engineering and few-shot learning
- Zero-shot classification

**Production & MLOps**
- Model deployment (FastAPI, Flask)
- Docker containerization
- Model monitoring and versioning
- Cloud deployment (basic)

**Specialized NLP Tasks**
- Named Entity Recognition (NER)
- Question Answering systems
- Text generation and summarization
- Cross-lingual NLP

---

**Tech Stack Overview:**
```python
{
    "expert": ["DistilBERT", "Sentence-Transformers", "multi-label classification"],
    "proficient": ["PyTorch", "Hugging Face", "text preprocessing", "scikit-learn"],
    "learning": ["GPT-family", "deployment", "LLM fine-tuning"]
}
```

### **Core Technical Stack**

**Programming & ML Frameworks**
- **Python** (Expert): PyTorch, TensorFlow, Keras, scikit-learn
- **NLP Libraries**: Hugging Face Transformers, spaCy, NLTK, Gensim
- **Data Science**: pandas, NumPy, matplotlib, seaborn, SciPy
- **R** (Proficient): tidyverse, text2vec, quanteda (text mining)
- **SQL**: Complex queries, data extraction, analysis

**Development & Tools**
- **Version Control**: Git, GitHub (collaborative workflows)
- **Environments**: Jupyter, Google Colab, VS Code
- **Cloud**: Google Colab Pro (GPU training), basic Docker

**Traditional NLP & Text Mining**
- **Text Representation**: TF-IDF, n-grams, bag-of-words, word embeddings
- **Classical ML**: Naive Bayes, SVM, Random Forest for text
- **Feature Engineering**: Text preprocessing, tokenization, stemming, lemmatization
- **Evaluation**: Precision, recall, F1-score, ROC-AUC, confusion matrices

**Deep Learning Beyond NLP**
- **Architectures**: CNNs, fully connected networks, RNNs (basic)
- **Techniques**: Transfer learning, regularization (dropout, batch norm), optimization
- **Applications**: Audio classification, EEG signal processing, image analysis

**Statistical Analysis & Research Methods**
- **Statistics**: Hypothesis testing, A/B testing, significance testing (t-tests, ANOVA)
- **Research Design**: Experimental design, data collection, reproducibility
- **Domain Applications**: Epidemiology, longitudinal analysis, cognitive neuroscience

### **Languages**
- 🇹🇷 Turkish (Native)
- 🇬🇧 English (Fluent - C2)
- 🇳🇱 Dutch (Intermediate - A2)
- 🇫🇷 French (Intermediate - B1.5)

*Multilingual background enhances understanding of cross-lingual NLP challenges*

---

## 📂 Featured NLP & ML Projects

### 🐱 [Cat Distress Detection — Audio ML & Welfare AI](https://github.com/pandabeliz/cat_meow_anlayser)
**Full ML pipeline for detecting distress in cat vocalisations | Fellowship Project**

Built an end-to-end machine learning system classifying cat meow recordings as **distress** (isolation) or **normal** (brushing/food-anticipation), deployed as a live web application. Developed as part of the **Electric Sheep Futurekind Fellowship** — an AI for animal protection programme.

**Technical Deep Dive:**
- Extracted **81 acoustic features** per recording: MFCCs (52), spectral descriptors (20), spectral entropy (2), temporal features (4), and pitch/F0 statistics (3)
- Trained and tuned a **LightGBM classifier** with Optuna (100 trials), optimising for **F2 score** — recall weighted 2× over precision, appropriate for welfare monitoring where missed distress is worse than a false alarm
- Identified a critical methodological failure in naive `train_test_split`: **19/21 cats leaked** between train and val, inflating recall to 88.6% while the per-cat AUC collapsed to 0.476 (worse than random) — the model was learning individual vocal signatures, not distress
- Corrected with **Leave-One-Cat-Out (LOCO) cross-validation**, producing honest generalisation results: **mean AUC 0.780 ± 0.248** across 16 evaluable cats
- Deployed via **Gradio on HuggingFace Spaces** as an inference API, with a **React frontend** (Lovable) for the user-facing interface

**Key Results:**
- 🎯 **Mean LOCO AUC: 0.780** — generalises to cats the model has never heard
- 🎯 **Decision threshold: 0.135** — tuned for welfare-appropriate recall
- 🎯 **11/16 evaluable cats** achieve AUC ≥ 0.80
- 🎯 **Live demo** processing real .wav uploads end-to-end

**Tech Stack:** Python · LightGBM · librosa · Optuna · Gradio · HuggingFace Spaces · React  
🔗 [GitHub](https://github.com/pandabeliz/cat_meow_anlayser) · [Live Demo](https://kitty-comm-analyzer.lovable.app/analyze) · [Model on HuggingFace](https://huggingface.co/belpekkan/cat_distress_detection)

---

### ♟️ [Chess Transformer Player — Seq2Seq Move Prediction](https://github.com/pandabeliz/UniversityProjects/tree/main/seq2seq_chess_T5)
**Fine-tuned T5-base transformer competing in a university chess tournament | INFOMTALC, Utrecht University**

Fine-tuned a T5-base encoder-decoder model to play chess by predicting moves from board state representations. Competed in a tournament against **Stockfish** and **Mistral-7B** as part of the INFOMTALC course at Utrecht University.

**Technical Deep Dive:**
- Framed chess as a **sequence-to-sequence task**: input is a FEN string augmented with all legal moves (`"chess: <FEN> legal: e2e4 d2d4 ..."`), output is the next move in UCI format
- Legal move augmentation via `python-chess` provides a strong validity signal — **0 illegal move fallbacks** in tournament play after this addition
- Fine-tuned **T5-base** on Lichess games filtered to Elo ≥ 1500, streamed via HuggingFace Datasets
- Training workflow: Google Colab GPU → `push_to_hub()` → reload from HuggingFace, eliminating fragile checkpoint management across sessions
- Evaluated against Stockfish (strong & weak) and Mistral-7B in a multi-player round-robin tournament

**Key Results:**
- 🎯 **0 illegal move fallbacks** in tournament play
- 🎯 Validation loss still decreasing at 3 epochs → continued training to 10–15 epochs planned
- 🎯 Model hosted and versioned on HuggingFace Hub for reproducibility

**Tech Stack:** Python · T5-base · HuggingFace Transformers · PyTorch · python-chess · Google Colab  
🔗 [GitHub](https://github.com/pandabeliz/UniversityProjects/tree/main/seq2seq_chess_T5) · [Model on HuggingFace](https://huggingface.co/belpekkan/chess_T5_seq2seq)

---

### 🏆 [Multi-Label Book Genre Classification Using Transformers](https://github.com/pandabeliz/BookGenrePrediction)
**Advanced NLP system with comprehensive model comparison**

Implemented state-of-the-art multi-label text classification comparing traditional methods against modern transformers across 54 genre categories.

**Technical Deep Dive:**
- Fine-tuned **DistilBERT** with custom multi-label classification head
- Compared 5 approaches: TF-IDF, Word2Vec (300-dim), FastText, DistilBERT, Sentence-Transformers
- Handled severe class imbalance using stratified sampling and weighted loss
- Implemented full statistical testing pipeline (paired t-tests across categories)
- Optimized hyperparameters using grid search and learning rate scheduling

**Key Results:**
- 🎯 **0.78 macro F1-score** with DistilBERT (15% improvement over TF-IDF baseline: 0.63)
- 🎯 **Sentence-Transformers excelled on rare genres** (0.81 F1 on classes with <100 examples)
- 🎯 **Strong generalization**: 0.72 F1 on held-out test set
- 🎯 **Statistical validation**: Transformer superiority confirmed across all 54 categories (p < 0.01)

**Tech Stack:** Python, PyTorch, Hugging Face Transformers, scikit-learn, pandas, matplotlib

**Why This Matters:** Demonstrates end-to-end NLP pipeline from data preprocessing through model deployment, with rigorous evaluation methodology applicable to any text classification problem.

---

### 🧠 [EEG Signal Classification - Auditory Language Processing](https://github.com/pandabeliz/UniversityProjects)
**Machine learning for cognitive neuroscience**

Classified EEG signals to distinguish neural responses to native vs. non-native language sounds, bridging cognitive science and machine learning.

**Approach:**
- Preprocessed neurophysiological data: filtering, artifact removal, temporal segmentation
- Compared traditional ML (Logistic Regression, SVM with multiple kernels) vs. deep learning (CNN)
- Designed 1D CNN architecture to capture temporal EEG patterns
- Applied rigorous cross-validation for robust performance estimation

**Results:**
- **78% accuracy with CNN** vs. 65% with best SVM (20% relative improvement)
- Demonstrated CNNs' ability to learn complex temporal patterns automatically
- Findings contribute to neurolinguistic research on bilingual language processing

**Tech Stack:** Python, TensorFlow, scikit-learn, MNE (EEG processing), NumPy

**Application:** Methods applicable to brain-computer interfaces, cognitive assessment, clinical diagnostics

---

### 🎵 [Deep Learning for Audio Emotion Prediction](https://github.com/pandabeliz/UniversityProjects)
**Neural networks for valence prediction from audio features**

Built fully connected neural network to predict emotional valence (positive/negative sentiment) from raw audio spectral features.

**Technical Implementation:**
- Designed 4-layer architecture with ReLU activation and batch normalization
- Applied dropout (0.3) for regularization to prevent overfitting
- Optimized hyperparameters using Bayesian optimization (Optuna framework)
- Extracted spectral features: MFCCs, spectral centroid, rolloff, zero-crossing rate

**Results:**
- Training MSE: 0.8713, **Validation MSE: 0.5068** (42% reduction, good generalization)
- Competitive performance vs. baseline CNN approaches
- Clean training curve demonstrates effective regularization

**Tech Stack:** TensorFlow, Keras, librosa (audio processing), Optuna, pandas

**Skills Demonstrated:** Neural network architecture design, regularization techniques, hyperparameter tuning

---

### 🐋 [Whale Migration Prediction Using Ensemble ML](https://github.com/pandabeliz/UniversityProjects)
**Environmental data science - climate change impact modeling**

Predicted blue whale migration pattern shifts due to climate change using ensemble machine learning methods.

**Approach:**
- Multi-output regression to predict migration routes across multiple coordinates
- Compared Linear Regression, Random Forest, and XGBoost for ensemble modeling
- Feature engineering from climate and oceanographic data
- Model interpretation to identify key environmental drivers

**Outcome:** Identified model limitations and improvement strategies, contributing to marine conservation efforts

**Tech Stack:** Python, XGBoost, Random Forest, scikit-learn, pandas

**Application:** Demonstrates ML for environmental sustainability and climate science

---


## 🎓 Education & Certifications

**Master of Science in Applied Data Science**  
Utrecht University | Expected June 2026  
*Key Courses:* Transformers: Applications in Language and Communication, Text and Media Analytics, Statistical Modeling, Human Network Analysis

**Bachelor of Science in Cognitive Science & Artificial Intelligence**  
Tilburg University | 2021-2024 | GPA: 7.15/10  
*Research Thesis:* Impact of Bilingualism on Native Listening Abilities (18 experiments, 250 participants, statistical analysis with mixed models)

**Professional Certifications:**
- 🏆 **Machine Learning Specialist Track** (DataCamp) - Supervised, unsupervised, deep learning, NLP, Spark ML
- 🏆 **Associate Data Scientist in Python** (DataCamp) - Data manipulation, visualization, ML workflows
- 🏆 **SQL Fundamentals** (DataCamp) - Database querying and data analysis

---

## 🔬 Research & Academic Experience

**Research Intern - Computational Linguistics**  
*Tilburg University | September 2023 - June 2025*

Conducted computational linguistics research investigating bilingual language processing using experimental methods and statistical analysis.

**Key Contributions:**
- Designed and executed **18 online experiments** with rigorous experimental controls
- Managed participant recruitment and data collection (**250 participants** across Turkish-English bilingual populations)
- Performed statistical analysis using **R (lme4, tidyverse)** with mixed effects models
- Applied hypothesis testing and ANOVA to identify significant cognitive factors
- Contributing to **manuscript preparation** for academic publication (in progress)

**Research Finding:** Age and specific audio cues significantly impact auditory recall accuracy (p < 0.05), while bilingualism showed minimal effects—contributing to theoretical understanding of cognitive load in language processing

**Skills Gained:** Experimental design, participant recruitment, statistical modeling, academic writing, data ethics

---

## 💼 What I'm Looking For

### 🎯 **Ideal Roles:**
- **NLP Engineer / Data Scientist** - Building production NLP systems
- **Machine Learning Engineer** - Deploying transformer models at scale
- **Research Scientist** - Applied NLP research in industry or academia
- **Data Scientist - Text Analytics** - Extracting insights from unstructured text

### 🌟 **Domains of Interest:**
- **Technology & SaaS:** Content moderation, search, recommendations, chatbots
- **Healthcare & Life Sciences:** Medical text mining, clinical NLP, patient data analysis
- **E-commerce & Retail:** Product categorization, sentiment analysis, customer insights
- **Financial Services:** Document analysis, risk assessment, compliance monitoring
- **Media & Publishing:** Content classification, topic modeling, automated tagging

### 💡 **What I Bring:**
- Deep expertise in **transformer fine-tuning** and modern NLP architectures
- Proven ability to **compare and evaluate** multiple modeling approaches systematically
- Strong foundation in both **classical ML and deep learning** for text
- Research experience with **rigorous statistical validation**
- Interdisciplinary perspective from **cognitive science** background
- **Multilingual** capabilities (4 languages) for cross-lingual applications

---


## 🌐 Connect With Me

[![Portfolio](https://img.shields.io/badge/Portfolio-belizpekkan.com-FF6B6B?style=for-the-badge&logo=google-chrome&logoColor=white)](https://belizpekkan.com)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Beliz_Pekkan-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/belizpekkan)
[![Email](https://img.shields.io/badge/Email-belizpekkan@gmail.com-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:belizpekkan@gmail.com)
[![GitHub](https://img.shields.io/badge/GitHub-pandabeliz-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/pandabeliz)

---

## 💡 Beyond the Code

- 📚 **Lifelong Learner:** Read **68 books** in 2025 - committed to continuous learning across AI, psychology, linguistics, and philosophy
- 🌍 **Multilingual NLP Enthusiast:** Native Turkish speaker, fluent in English, learning Dutch (A2) and French (B1.5) - brings unique perspective to cross-lingual NLP challenges
- 🎨 **Creative Side:** Freelance website designer since 2020 - built **5+ custom websites** across finance, healthcare, and hospitality
- 🧠 **Cognitive Science Foundation:** Background in human cognition informs user-centered approach to AI design
- 🌱 **Passionate About:** Applying NLP to meaningful problems - environmental sustainability, healthcare accessibility, education technology

---

## 📈 2026 Goals

- [ ] Complete Master's thesis on advanced NLP application (transformers/LLMs)
- [ ] Publish **6-8 technical blog posts** on transformers, NLP, and text analytics
- [ ] Contribute to **open-source NLP projects** (Hugging Face, spaCy, or text mining libraries)
- [ ] Build and deploy a **production-ready NLP API** (sentiment analysis or classification service)
- [ ] Achieve **Dutch B2** language proficiency
- [ ] Secure **NLP/ML Engineer or Data Scientist** position in Netherlands
- [ ] Present research at **NLP conference or meetup** (PyData, NLP Summit, ACL)

---

<div align="center">

### "Building intelligent NLP systems that understand language, context, and meaning"

![Profile Views](https://komarev.com/ghpvc/?username=pandabeliz&color=blueviolet&style=flat-square)

</div>

---

*Last updated: March 2026*
