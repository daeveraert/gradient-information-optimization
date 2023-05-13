from sentence_transformers import SentenceTransformer
import time



class GenerateEmbeddings:
    def __init__(self, model_name, device='cuda'):
        self.model = SentenceTransformer(model_name)
        self.device = device

    def generate_embeddings(self, input_file_path, output_file_path):
        """Generate Embeddings from a text file
        :param input_file_path: path to input text, one sentence per line
        :param output_file_path: path to desired output file
        """
        print('Reading File...')
        with open(input_file_path, 'r') as fp:
            sentences = fp.readlines()
        print('Generating Embeddings... This May Take a While')
        start = time.time()
        embeddings = self.model.encode(sentences, device=self.device)
        end = time.time()

        print("Time Taken (s): " + str(end - start))

        print("Writing Embeddings.. This May Take a While")
        with open(output_file_path, 'w') as op:
            for i, each in enumerate(embeddings):
                op.write(str(each.tolist()).strip() + "\n")
