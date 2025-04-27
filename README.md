# **Chat with Your Video Library**
*CS-GY 6613 AI Project*

Refer to the detailed documentation above - "Artificial Intelligence Project Report"
### Team members:

| Name | NetID | 
|----------|----------|
| Swapnil Sharma | ss19753 |
| Sindhu Jyoti Dutta | sd6201 |


### Architecture:

<img width="554" alt="Screenshot 2025-04-27 at 1 52 39 PM" src="https://github.com/user-attachments/assets/57e00c9e-1dbf-4b6f-9ebc-1a1408c4da7b" />

### Setup instructions:
1. Download the below "youtube_dataset.tar" dataset from HuggingFace and place it in  `/datasets/.` :
   
   `https://huggingface.co/datasets/aegean-ai/ai-lectures-spring-24/tree/main`
2. Run the services of the docker compose file in the following order:
    - mongo
    - mongo-express
    - video-etl
    - qdrant
    - embedder
    - retriever-and-response-generator
      
  ![image](https://github.com/user-attachments/assets/c8ffafb6-e5a6-47da-b458-84f71eb11cc4)

3. Access the Gradio app : `http://localhost:7860/`
   
![image](https://github.com/user-attachments/assets/ee23c2bd-cae8-45d1-a922-bdde51e1dfcd)


     
