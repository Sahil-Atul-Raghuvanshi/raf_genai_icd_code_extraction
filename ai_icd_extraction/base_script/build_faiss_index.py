"""
One-time script to build FAISS index from ICD-10 codes.
Run this once to create the index, then reuse it in the app.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

def build_icd_faiss_index():
    """
    Build FAISS index from ICD-10 CSV file.
    Creates embeddings for all ICD code descriptions.
    """
    
    print("Loading ICD-10 codes...")
    # Update path to use relative path from script location
    data_dir = Path(__file__).resolve().parent.parent / "data"
    df = pd.read_csv(data_dir / "icd10cm_2026.csv", dtype=str)
    
    # Filter to only billable codes (optional - for better quality)
    # Uncomment the line below to only index billable codes
    # df = df[df["is_billable"] == "1"]
    
    # Prepare texts and metadata
    print(f"Processing {len(df)} ICD codes...")
    
    texts = []
    metadatas = []
    
    for _, row in df.iterrows():
        # Combine code and description for better context
        text = f"{row['code']}: {row['long_title']}"
        texts.append(text)
        
        metadata = {
            "code": row["code"],
            "long_title": row["long_title"],
            "short_title": row["short_title"],
            "is_billable": row["is_billable"]
        }
        metadatas.append(metadata)
    
    # Initialize embedding model (using local Hugging Face model - no API needed)
    print("Initializing embedding model (downloading if first time)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",  # Fast, efficient model
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Build FAISS index with batching
    print("Building FAISS index (this may take 10-15 minutes)...")
    print("Processing in batches for better memory management...")
    
    batch_size = 1000  # Larger batches since we're using local embeddings
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    all_indices = []
    
    for i in range(0, len(texts), batch_size):
        batch_num = i // batch_size + 1
        print(f"Processing batch {batch_num}/{total_batches}...")
        
        batch_texts = texts[i:i + batch_size]
        batch_metadatas = metadatas[i:i + batch_size]
        
        try:
            # Create FAISS index for this batch
            batch_index = FAISS.from_texts(
                texts=batch_texts,
                embedding=embeddings,
                metadatas=batch_metadatas
            )
            
            all_indices.append(batch_index)
            
            # Small delay for system resources
            if i + batch_size < len(texts):
                import time
                time.sleep(1)  # 1 second delay between batches
                
        except Exception as e:
            print(f"Error processing batch {batch_num}: {e}")
            print("Waiting 30 seconds before retrying...")
            import time
            time.sleep(30)
            
            # Retry the batch
            try:
                batch_index = FAISS.from_texts(
                    texts=batch_texts,
                    embedding=embeddings,
                    metadatas=batch_metadatas
                )
                all_indices.append(batch_index)
            except Exception as e2:
                print(f"Failed to process batch {batch_num} after retry: {e2}")
                print("Skipping this batch and continuing...")
                continue
    
    # Merge all batch indices
    print("\nMerging all batches into single index...")
    faiss_index = all_indices[0]
    for idx in all_indices[1:]:
        faiss_index.merge_from(idx)
    
    # Save to disk
    index_path = data_dir / "faiss_icd_index"
    os.makedirs(index_path, exist_ok=True)
    
    print(f"Saving index to {index_path}...")
    faiss_index.save_local(str(index_path))
    
    print(f"\n✅ FAISS index built successfully!")
    print(f"   Total codes indexed: {len(texts)}")
    print(f"   Index location: {index_path}")
    print(f"\nYou can now use this index in your application.")

if __name__ == "__main__":
    build_icd_faiss_index()
