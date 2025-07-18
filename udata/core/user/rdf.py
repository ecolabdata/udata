"""
This module centralize user helpers for RDF/DCAT serialization and parsing
"""

from rdflib import BNode, Graph, Literal, URIRef
from rdflib.namespace import FOAF, RDF, RDFS

from udata.rdf import namespace_manager


def user_to_rdf(user, graph=None):
    """
    Map a Resource domain model to a DCAT/RDF graph
    """
    graph = graph or Graph(namespace_manager=namespace_manager)
    if user.id:
        id = URIRef(user.url_for(_useId=True))
    else:
        id = BNode()
    o = graph.resource(id)
    o.set(RDF.type, FOAF.Person)
    o.set(FOAF.name, Literal(user.fullname))
    o.set(RDFS.label, Literal(user.fullname))
    if user.website:
        o.set(FOAF.homepage, URIRef(user.website))
    return o
