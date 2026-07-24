The current RAG pipeline uses fixed-size text chunking with overlap and dense vector retrieval. While this works well for short documents, answer quality may decrease when working with larger documents.

To improve retrieval performance, I would consider semantic chunking instead of fixed-size chunking, as it preserves the meaning of paragraphs. I would also use hybrid search by combining dense vector search with keyword-based retrieval (BM25), which helps when users search for exact terms.

Another improvement would be adding a re-ranking model (Cross Encoder) to reorder the retrieved chunks before passing them to the language model. Finally, using stronger embedding models and metadata filtering could further improve retrieval accuracy and reduce irrelevant context.