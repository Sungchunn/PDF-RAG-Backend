# RAG: Science and Math

This document explains how retrieval-augmented generation (RAG) works in this backend and the key math used in retrieval.

## Pipeline summary

1. **Parse PDF into blocks** with bounding boxes and line numbers.
2. **Embed each block** into a vector space (OpenAI or Gemini embeddings).
3. **Store embeddings in pgvector** (`document_chunks.embedding`).
4. **Embed the query** into the same vector space.
5. **Retrieve top-K blocks** by cosine similarity.
6. **Generate an answer** using the LLM with retrieved context.

## Embeddings

- Each text block is embedded via LlamaIndex wrappers:
  - `OpenAIEmbedding` or `GeminiEmbedding`.
- The resulting vector is stored as `document_chunks.embedding`.
- The migration sets the vector dimension to 1536 (OpenAI `text-embedding-3-small`).
- If you switch embedding models, ensure the DB vector dimension matches the model output size.

## Similarity search math

pgvector uses cosine distance via the `<=>` operator. The query converts distance to a similarity score:

```
similarity = 1 - (embedding <=> query_embedding)
```

Cosine similarity for vectors `u` and `v` is:

```
cos_sim(u, v) = (u · v) / (||u|| * ||v||)
```

Higher similarity means closer semantic meaning. Retrieval returns the top `settings.similarity_top_k` chunks.

## Prompt construction

Retrieved blocks are joined into a single context string with source markers:

```
[Source: Page P, Lines L1-L2]
<block text>
```

The final prompt template:

```
Based on the following context from documents, answer the question.

Context:
<context blocks>

Question: <user question>

Answer:
```

## Chunking behavior

- The RAG engine uses PDF text blocks as chunks (one embedding per block).
- `settings.chunk_size` and `settings.chunk_overlap` configure LlamaIndex defaults, but the current pipeline does not apply an additional sentence split.

## Citations and alignment

- Each retrieved block carries its page number, bounding box, and line range.
- These fields are returned to the client for precise highlighting.
