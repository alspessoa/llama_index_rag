import yaml
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def load_config_from_file(file_path):
    try:
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
            return config
    except Exception as e:
        print(f"Ocorreu um erro ao carregar o arquivo de configuração: {e}")
        return None


def configure_settings(settings=Settings):
    config_file_path = "rag/config.yaml"
    config = load_config_from_file(config_file_path)

    if config is not None:
        settings.embed_model = HuggingFaceEmbedding(
            model_name=config['llama_index']['embeddings']['model_name'])
        settings.llm = eval(config['llama_index']['llm'])
        settings.chunk_size = config['llama_index']['chunk_size']
        settings.chunk_overlap = config['llama_index']['chunk_overlap']

    return settings
