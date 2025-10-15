from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY=os.environ.get('pcsk_4eEpPR_F6wwRVzB8TgYCy3fWRoU4YZzurm25TPFvumfU824hA6VuZ8cPGrK6Tuwe33iTK9')
GOOGLE_API_KEY =os.environ.get('AIzaSyALhnEJg2hwwkoc3uiyKC-4C-4wfZ6dHM0')

extracted_data=load_pdf_file(data='Data/')
text_chunks=text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

pc = Pinecone(api_key="pcsk_4eEpPR_F6wwRVzB8TgYCy3fWRoU4YZzurm25TPFvumfU824hA6VuZ8cPGrK6Tuwe33iTK9")

index_name = "medicalbot"


pc.create_index(
    name=index_name,
    dimension=384, 
    metric="cosine", 
    spec=ServerlessSpec(
        cloud="aws", 
        region="us-east-1"
    ) 
) 