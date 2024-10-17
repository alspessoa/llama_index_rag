from glob import glob

from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.llms.ollama import Ollama
from llama_parse import LlamaParse

from rag.settings import configure_settings


class RAGConfig:
    """Configuration class for the RAG model."""

    load_dotenv()
    configure_settings()

    def __init__(self,
                 top_k: int = 2,
                 llm_model: str = "llama3.1:latest",
                 request_timeout: float = 120.0,
                 similarity_cutoff: float = 0.5):
        self.top_k = top_k
        self.llm_model = llm_model
        self.request_timeout = request_timeout
        self.similarity_cutoff = similarity_cutoff


class RAG:
    """Class for interacting with the RAG model."""

    def __init__(self, raw_documents_path: str):
        self.config = RAGConfig()
        self.raw_documents_path = raw_documents_path
        self.llm = Ollama(model=self.config.llm_model,
                          request_timeout=self.config.request_timeout)

    def get_parser(self, result_type: str = 'markdown') -> LlamaParse:
        """Get a parser instance."""
        return LlamaParse(result_type=result_type)

    def get_documents_from_source(self) -> dict:
        """Load documents from the source directory."""
        parser = self.get_parser()
        file_extractor = {'.pdf': parser}
        documents = SimpleDirectoryReader(
            input_files=glob(f'{self.raw_documents_path}/*'),
            file_extractor=file_extractor).load_data()
        return documents

    def get_index(self) -> VectorStoreIndex:
        """Get the index instance."""
        documents = self.get_documents_from_source()
        index = VectorStoreIndex.from_documents(documents)
        return index

    def retriever(self, index: VectorStoreIndex) -> VectorIndexRetriever:
        """Get a retriever instance."""
        retriever = VectorIndexRetriever(index=index,
                                         similarity_top_k=self.config.top_k)
        return retriever

    def query_engine(self,
                     retriever: VectorIndexRetriever) -> RetrieverQueryEngine:
        """Get a query engine instance."""
        node_postprocessors = [
            SimilarityPostprocessor(
                similarity_cutoff=self.config.similarity_cutoff)
        ]
        query_engine = RetrieverQueryEngine(
            retriever=retriever, node_postprocessors=node_postprocessors)
        return query_engine

    def get_most_similar_content(self, query: str) -> dict:
        """Get the most similar content for a given query."""
        index = self.get_index()
        retriever = self.retriever(index)
        query_engine = self.query_engine(retriever)
        response = query_engine.query(query)
        return response

    def get_rag_answer(self, query: str, comment: str) -> str:
        """Get the RAG answer for a given query and comment."""
        similar_content = self.get_most_similar_content(query)
        context = generate_context(self.config.top_k,
                                   response=similar_content,
                                   comment=comment)
        return self.llm.complete(context)


def generate_context(top_k: int, response: dict, comment: str) -> str:
    """Generate the context for the LLAMA model."""

    def _prompt_template(context, comment):
        prompt_template_w_context = f"""[INST]RAGSolution, functioning as a virtual data science consultant, communicates in clear, accessible language, escalating to technical depth upon request. \
        It reacts to feedback aptly and ends responses with its signature '-RAGSolution'. \
        RAGSolution will tailor the length of its responses to match the viewer's comment, providing concise acknowledgments to brief expressions of gratitude or feedback, \
        thus keeping the interaction natural and engaging.
        
        If RAGSolution CANNOT answer a question or DOESN'T find proper information, RAGSolution will simple respond that it doesn't know.

        **Context:** {context}
        
        **Comment to Respond To:**
        
        {comment}
        
        **Desired Response Type:**
        
        1. Concise Acknowledgment (e.g., "Thank you for sharing!")
        2. Brief Explanation (e.g., "I'm glad you found that helpful.")
        3. In-Depth Explanation (e.g., "Let me provide more context on [topic].")
        4. Alternative Solution or Resource (e.g., "Have you considered trying [alternative approach]?")
        
        **Additional Context:**
        
        * Relevant knowledge sources:
            + Company wiki
            + Industry reports
            + Academic research papers
        * Key concepts to address:
            + Data science fundamentals
            + Machine learning techniques
            + Statistical analysis methods
        
        Please respond to the comment using the context and desired response type above. If you need more information or clarification, feel free to ask.
        
        -RAGSolution
        
        [/INST]
         """

        return prompt_template_w_context

    context = "Context:\n"

    for i in range(top_k):
        context = context + response.source_nodes[i].text + "\n\n"

    return _prompt_template(context, comment)
