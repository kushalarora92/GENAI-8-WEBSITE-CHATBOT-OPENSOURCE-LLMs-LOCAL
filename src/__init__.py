from langchain_community.document_loaders import UnstructuredURLLoader

URLs = [
    "https://www.nubosushi.com/",
    "https://www.nubosushi.com/food",
    "https://www.nubosushi.com/drink",
    "https://www.nubosushi.com/contact"
]

loaders = UnstructuredURLLoader(URLs)
documents = loaders.load()
