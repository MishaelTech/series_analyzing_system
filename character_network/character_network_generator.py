import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from pyvis.network import Network


class CharacterNetworkGenerator:
    def __init__(self):
        pass

    # Occurence per character pair
    def generate_character_network(self,df):

        window=10 # if two charater appear in 10 sentences then we increment the counter
        entity_relationship=[]

        for row in df['ners']:
            previous_entity_in_window=[]

            # Loop through each sentence
            for sentence in row:
                previous_entity_in_window.append(list(sentence)) # Its a list of list (2d list)
                previous_entity_in_window=previous_entity_in_window[-window:] # last 10 sentences

                # Flatten 2d list into 1d list
                previous_entities_flatten=sum(previous_entity_in_window,[])

                # Loop through the entity in the sentence and over the entities in the window
                for entity in sentence:
                    for entity_in_window in previous_entities_flatten:
                        if entity!=entity_in_window:
                            entity_relationship.append(sorted([entity,entity_in_window]))

        relationship_df=pd.DataFrame({"value": entity_relationship})
        relationship_df["source"]=relationship_df["value"].apply(lambda x: x[0])
        relationship_df["target"]=relationship_df["value"].apply(lambda x: x[1])
        relationship_df=relationship_df.groupby(["source","target"]).count().reset_index()
        # Sorting to make must character that appear are on the top and less character are at the bottom
        relationship_df=relationship_df.sort_values(by="value",ascending=False)

        return relationship_df


    def draw_network_graph(self, relationship_df):
        # Taking the data and creating a character network
        # sort and take the first 200 Number of occurances
        relationship_df=relationship_df.sort_values(by="value",ascending=False)
        relationship_df=relationship_df.head(200)

        # Transforming it to a network using networkx library
        G=nx.from_pandas_edgelist(
            relationship_df,
            source="source",
            target="target",
            edge_attr="value",
            create_using=nx.Graph()
        )

        net = Network(notebook=True, width="1000px", height="700px", bgcolor="#222222", font_color="white",cdn_resources="remote")
        node_degree = dict(G.degree)

        nx.set_node_attributes(G, node_degree, "size")
        net.from_nx(G)
        #net.show("character_network.html")
        # save it
        #net.save_graph("character_network.html")

        # Rather than saving we get it
        html = net.generate_html()

        # Cleaning Html: In every single comma we have double quotation to not break the code
        html=html.replace("'"," \"")

        # An iframe to store the html
        output_html = f"""<iframe style="width: 100%; height: 600px;margin:0 auto" name="result" allow="midi; geolocation; microphone; camera;
    display-capture; encrypted-media;" sandbox="allow-modals allow-forms
    allow-scripts allow-same-origin allow-popups
    allow-top-navigation-by-user-activation allow-downloads" allowfullscreen=""
    allowpaymentrequest="" frameborder="0" srcdoc='{html}'></iframe>"""

        return output_html