### Dissertation Title

**A Personalized Media Recommender System with Adaptive Learning for Mobile Streaming Platforms**

---

### Context and Motivation

In the dynamic landscape of digital streaming, platforms like Netflix and YouTube continuously aim to retain user engagement by aligning content recommendations with rapidly evolving viewer preferences. These shifts are driven by factors such as multilingual content demand and niche audience interests, presenting a challenge for traditional, largely static recommendation systems. These conventional models often lack the flexibility to accommodate diverse user needs and changing behaviors effectively, particularly in mobile applications where user attention spans are short. This project introduces an adaptive recommendation model designed for mobile streaming platforms, which incorporates reinforcement learning for real-time learning and user alignment, improving recommendation relevance over time.

### Problem Statement

Static recommendation systems frequently fail to capture ongoing shifts in user preferences, resulting in potential dissatisfaction and disengagement. This dissertation addresses the shortcomings of static algorithms by developing a model that dynamically responds to real-time user feedback, employing reinforcement learning to adjust recommendations on-the-fly. Through this approach, the system bridges the gap between static recommendation paradigms and dynamic adaptability, positioning itself as a responsive solution capable of better meeting evolving user needs.

### Methodology and Key Algorithms

#### 1. Matrix Factorization Models
Matrix Factorization, with techniques such as Singular Value Decomposition (SVD) and Alternating Least Squares (ALS), reduces large, sparse user-item matrices into lower-dimensional representations, revealing latent factors that influence preferences. This technique provides a robust basis for preference prediction by identifying the deeper structures within user interactions.
- **Surprise Library** ([GitHub](https://github.com/NicolasHug/Surprise)): An excellent resource for collaborative filtering and matrix factorization, with ready implementations of SVD and ALS.
- **Implicit ALS** ([GitHub](https://github.com/benfred/implicit)): Focused on implicit data matrix factorization, useful for cases with implicit feedback (e.g., clicks, views).

#### 2. Collaborative Filtering
Collaborative Filtering (CF) uses similarity metrics between users or items to generate recommendations. User-based CF recommends items that similar users have enjoyed, while item-based CF suggests items akin to those the user has previously liked. By employing both approaches, the model can build upon known user-item associations and adapt to shifting preferences through periodic retraining.
- **LightFM Library** ([GitHub](https://github.com/lyst/lightfm)): A hybrid recommendation library that combines collaborative filtering with content-based methods, ideal for handling diverse preferences and niche recommendations.

#### 3. Reinforcement Learning (RL)
Reinforcement Learning introduces dynamic adaptation by continually learning from user interactions. Techniques like Q-learning or Deep Q-Networks (DQN) enable the model to refine recommendations in real-time based on feedback, creating a core adaptive layer.
- **PyTorch DQN Tutorial** ([GitHub](https://github.com/philtabor/Youtube-Code-Repository/tree/master/ReinforcementLearning/DeepQLearning)): Provides a foundational implementation for building adaptive recommendation models with reinforcement learning.

### Datasets for Training and Evaluation

The project will leverage two main datasets to test and validate the adaptive recommendation model:

- **Movielens 100k Dataset** ([Movielens](https://grouplens.org/datasets/movielens/100k/)): With 100,000 ratings and metadata, this dataset is well-suited for testing matrix factorization and collaborative filtering models.
- **Douban Dataset** ([Douban Dataset Repository](https://paperswithcode.com/dataset/douban)): A dataset reflecting user interactions within a Chinese-speaking community, allowing for tests of the model’s ability to accommodate niche preferences and minority languages.

**Schedule**

In the first week, the focus will be on finalizing the initial code and setting up the project’s GitHub repository, where baseline algorithms, dataset configurations, and a detailed project description will be uploaded. Early testing of basic recommendation algorithms on selected datasets, such as Movielens 100k and Douban, will provide preliminary data, laying a foundation for the adaptive model.

The second week will involve refining the adaptive model by incorporating insights from initial testing and implementing reinforcement learning components. Initial validation will begin using key user interaction metrics, such as engagement frequency and click-through rates, to gauge the model's effectiveness and identify areas for further improvement.

In the third week, complete model training, validation, and testing will be conducted, generating results that can be used for performance comparison. Documentation of the training process and visual results will be collected to support the analysis and display in the Results section of the dissertation.

The fourth week will focus on ablation studies, assessing the impact of hyperparameter adjustments, such as learning rate and the balance between exploration and exploitation, to optimize the model’s adaptive performance. All visualizations and findings will be finalized in this stage, along with the preparation of detailed data displays for the dissertation.

By the fifth and final week, the project’s code will be thoroughly reviewed and organized in the GitHub repository, with complete documentation and a README detailing the algorithms, datasets, and system architecture. The dissertation will integrate all final results and visuals, and the full draft will be completed, providing an in-depth presentation of the system’s development, performance, and insights gained from this adaptive approach to media recommendation.

### **Preliminary references**

- **"Deep Learning based Recommender System: A Survey and New Perspectives"**  
  Overview of deep learning applications in recommendation systems, including collaborative filtering and reinforcement learning.
  - [Link to Paper](https://paperswithcode.com/paper/deep-learning-based-recommender-system-a)
  - [Code Implementations](https://paperswithcode.com/paper/deep-learning-based-recommender-system-a)

- **"Neural Collaborative Filtering"**  
  This paper presents a neural architecture for collaborative filtering, which may serve as an advanced method for user-item interaction modeling.
  - [Link to Paper](https://paperswithcode.com/paper/neural-collaborative-filtering)
  - [Code Implementations](https://paperswithcode.com/paper/neural-collaborative-filtering)

- **"Variational Autoencoders for Collaborative Filtering"**  
  A deep learning-based approach leveraging variational autoencoders to improve collaborative filtering.
  - [Link to Paper](https://paperswithcode.com/paper/variational-autoencoders-for-collaborative)
  - [Code Implementations](https://paperswithcode.com/paper/variational-autoencoders-for-collaborative)

- **"Matrix Factorization Techniques for Recommender Systems"**  
  A foundational paper detailing matrix factorization techniques for recommendation, including SVD and ALS.
  - [Link to Paper](https://paperswithcode.com/paper/matrix-factorization-techniques-for)
  - [Code Implementations](https://paperswithcode.com/paper/matrix-factorization-techniques-for)

- **"Collaborative Denoising Auto-Encoders for Top-N Recommender Systems"**  
  This paper focuses on enhancing collaborative filtering using autoencoders, complementing matrix factorization methods.
  - [Link to Paper](https://paperswithcode.com/paper/collaborative-denoising-auto-encoders-for-top)
  - [Code Implementations](https://paperswithcode.com/paper/collaborative-denoising-auto-encoders-for-top)

- **"Reinforcement Learning for Recommender Systems: A Case Study"**  
  This paper explores reinforcement learning applied to recommendation systems, detailing approaches like DQN.
  - [Link to Paper](https://paperswithcode.com/paper/reinforcement-learning-for-recommender)
  - [Code Implementations](https://paperswithcode.com/paper/reinforcement-learning-for-recommender)

- **"SlateQ: A Tractable Decomposition for Reinforcement Learning Recommender Systems"**  
  SlateQ is a reinforcement learning framework designed for large-scale recommendation tasks, especially valuable for handling extensive user-item matrices.
  - [Link to Paper](https://paperswithcode.com/paper/slateq-a-tractable-decomposition-for)
  - [Code Implementations](https://paperswithcode.com/paper/slateq-a-tractable-decomposition-for)

- **"DQN with Prioritized Experience Replay"**  
  While not exclusively for recommendation, this paper on DQN with Prioritized Experience Replay enhances adaptive learning by selectively replaying significant experiences, ideal for dynamic preference modeling.
  - [Link to Paper](https://paperswithcode.com/paper/prioritized-experience-replay)
  - [Code Implementations](https://paperswithcode.com/paper/prioritized-experience-replay)

- **"Multi-Objective Reinforcement Learning for Recommender Systems"**  
  This paper discusses a hybrid approach using multi-objective reinforcement learning to balance multiple goals in recommendations, which can enhance user satisfaction and engagement.
  - [Link to Paper](https://paperswithcode.com/paper/multi-objective-reinforcement-learning-for)
  - [Code Implementations](https://paperswithcode.com/paper/multi-objective-reinforcement-learning-for)

- **"Deep Reinforcement Learning for List-wise Recommendations"**  
  Explores list-wise recommendation strategies, suitable for streaming platforms with sequential content displays.
  - [Link to Paper](https://paperswithcode.com/paper/deep-reinforcement-learning-for-list-wise)
  - [Code Implementations](https://paperswithcode.com/paper/deep-reinforcement-learning-for-list-wise)
