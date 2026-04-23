import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from elasticsearch import Elasticsearch
from elasticsearch.exceptions import NotFoundError
from elasticsearch_dsl import (
    Date,
    Document,
    Float,
    Integer,
    Keyword,
    Text,
    analyzer,
    query,
    token_filter,
    tokenizer,
)
from elasticsearch_dsl.connections import connections

from udata_search_service.entities import (
    Dataservice,
    Dataset,
    Discussion,
    Organization,
    Post,
    Reuse,
    Topic,
)

log = logging.getLogger(__name__)


@dataclass
class TermsFacet:
    name: str
    es_field: str


@dataclass
class DateRangeFacet:
    name: str
    es_field: str


DATE_RANGES = [
    {"key": "last_30_days", "from": "now-30d/d"},
    {"key": "last_12_months", "from": "now-12M/d"},
    {"key": "last_3_years", "from": "now-3y/d"},
]


def _parse_filtered_facets(aggregations, facets: list) -> dict:
    """Parse ES aggregations built with the filter-wrapper pattern into a facets dict."""
    result = {}
    for facet in facets:
        if isinstance(facet, TermsFacet):
            filtered_name = f"{facet.name}_filtered"
            total_name = f"{facet.name}_total"
            if hasattr(aggregations, filtered_name):
                fa = getattr(aggregations, filtered_name)
                if hasattr(fa, facet.name):
                    buckets = [
                        {"name": b.key, "count": b.doc_count}
                        for b in getattr(fa, facet.name).buckets
                    ]
                    total = int(fa.total.value) if hasattr(fa, "total") else 0
                    result[facet.name] = [{"name": "all", "count": total}] + buckets
            elif hasattr(aggregations, facet.name):
                buckets = [
                    {"name": b.key, "count": b.doc_count}
                    for b in getattr(aggregations, facet.name).buckets
                ]
                total = (
                    int(getattr(aggregations, total_name).value)
                    if hasattr(aggregations, total_name)
                    else 0
                )
                result[facet.name] = [{"name": "all", "count": total}] + buckets
        elif isinstance(facet, DateRangeFacet):
            if hasattr(aggregations, "last_update_filtered"):
                fa = aggregations.last_update_filtered
                buckets = [{"name": b.key, "count": b.doc_count} for b in fa.last_update.buckets]
                total = int(fa.total.value) if hasattr(fa, "total") else 0
                result["last_update"] = [{"name": "all", "count": total}] + buckets
            elif hasattr(aggregations, "last_update"):
                buckets = [
                    {"name": b.key, "count": b.doc_count} for b in aggregations.last_update.buckets
                ]
                total = (
                    int(aggregations.last_update_total.value)
                    if hasattr(aggregations, "last_update_total")
                    else 0
                )
                result["last_update"] = [{"name": "all", "count": total}] + buckets
    return result


SEARCH_SYNONYMS = [
    "AMD, administrateur ministériel des données, AMDAC",
    "lolf, loi de finance",
    "waldec, RNA, répertoire national des associations",
    "ovq, baromètre des résultats",
    "contour, découpage",
    "rp, recensement de la population",
]

# French analyzer based on https://jolicode.com/blog/construire-un-bon-analyzer-francais-pour-elasticsearch
french_elision = token_filter(
    "french_elision",
    type="elision",
    articles_case=True,
    articles=["l", "m", "t", "qu", "n", "s", "j", "d", "c", "jusqu", "quoiqu", "lorsqu", "puisqu"],
)
french_stop = token_filter("french_stop", type="stop", stopwords="_french_")
french_stemmer = token_filter("french_stemmer", type="stemmer", language="light_french")
french_synonym = token_filter(
    "french_synonym", type="synonym", ignore_case=True, expand=True, synonyms=SEARCH_SYNONYMS
)


dgv_analyzer = analyzer(
    "french_dgv",
    tokenizer=tokenizer("icu_tokenizer"),
    filter=["icu_folding", french_elision, french_synonym, french_stemmer, french_stop],
)


class IndexDocument(Document):
    @classmethod
    def _matches(cls, hit):
        # ES returns the physical index name in hits (e.g. "udata-dataset-2024-01-01-12-00"),
        # not the alias ("udata-dataset"). Default _matches uses fnmatch exact match which fails.
        return hit.get("_index", "").startswith(cls._index._name)

    @classmethod
    def init_index(cls, es_client: Elasticsearch, suffix: str) -> None:
        alias = cls._index._name
        pattern = alias + "-*"

        log.info(f"Saving template {alias} on the following pattern: {pattern}")
        index_template = cls._index.as_template(alias, pattern)
        index_template.save()

        if not cls._index.exists():
            log.info(f"Creating index {alias + suffix}")
            es_client.indices.create(index=alias + suffix)
            es_client.indices.put_alias(index=alias + suffix, name=alias)
        else:
            log.info(f"Index on alias {alias} already exists")

    @classmethod
    def delete_indices(cls, es_client: Elasticsearch) -> None:
        pattern = cls._index._name + "*"
        log.info(f"Deleting indices with pattern {pattern}")
        es_client.indices.delete(index=pattern)


class SearchableDataservice(IndexDocument):
    class Index:
        name = "dataservice"

    title = Text(analyzer=dgv_analyzer)
    created_at = Date()
    metadata_modified_at = Date()
    tags = Keyword(multi=True)
    topics = Keyword(multi=True)
    badges = Keyword(multi=True)
    organization = Keyword()
    description = Text(analyzer=dgv_analyzer)
    organization_name = Text(analyzer=dgv_analyzer, fields={"keyword": Keyword()})
    organization_with_id = Keyword()
    owner = Keyword()
    views = Float()
    followers = Float()
    description_length = Float()
    access_type = Keyword()
    producer_type = Keyword(multi=True)
    documentation_content = Text(analyzer=dgv_analyzer)


class SearchableTopic(IndexDocument):
    class Index:
        name = "topic"

    name = Text(analyzer=dgv_analyzer, fields={"keyword": Keyword()})
    description = Text(analyzer=dgv_analyzer)
    tags = Keyword(multi=True)
    featured = Integer()
    private = Integer()
    created_at = Date()
    last_modified = Date()
    organization = Keyword()
    organization_name = Text(analyzer=dgv_analyzer, fields={"keyword": Keyword()})
    organization_with_id = Keyword()
    producer_type = Keyword(multi=True)
    nb_datasets = Integer()
    nb_reuses = Integer()
    nb_dataservices = Integer()


class SearchableDiscussion(IndexDocument):
    class Index:
        name = "discussion"

    title = Text(analyzer=dgv_analyzer)
    content = Text(analyzer=dgv_analyzer)
    created_at = Date()
    closed_at = Date()
    closed = Integer()
    subject_class = Keyword()
    subject_id = Keyword()


class SearchablePost(IndexDocument):
    class Index:
        name = "post"

    name = Text(analyzer=dgv_analyzer)
    headline = Text(analyzer=dgv_analyzer)
    content = Text(analyzer=dgv_analyzer)
    tags = Keyword(multi=True)
    created_at = Date()
    last_modified = Date()
    published = Date()


class SearchableOrganization(IndexDocument):
    class Index:
        name = "organization"

    name = Text(analyzer=dgv_analyzer)
    acronym = Text()
    description = Text(analyzer=dgv_analyzer)
    url = Text()
    orga_sp = Integer()
    created_at = Date()
    followers = Float()
    views = Float()
    reuses = Float()
    datasets = Integer()
    badges = Keyword(multi=True)
    producer_type = Keyword(multi=True)


class SearchableReuse(IndexDocument):
    class Index:
        name = "reuse"

    title = Text(analyzer=dgv_analyzer)
    url = Text()
    created_at = Date()
    last_modified = Date()
    archived = Date()
    orga_followers = Float()
    views = Float()
    followers = Float()
    datasets = Integer()
    featured = Integer()
    type = Keyword()
    topic = Keyword()  # Metadata topic (health, transport, etc.)
    topic_object = Keyword(multi=True)  # Topic objects linked via TopicElement
    tags = Keyword(multi=True)
    badges = Keyword(multi=True)
    organization = Keyword()
    description = Text(analyzer=dgv_analyzer)
    organization_name = Text(analyzer=dgv_analyzer, fields={"keyword": Keyword()})
    organization_with_id = Keyword()
    organization_badges = Keyword(multi=True)
    owner = Keyword()
    producer_type = Keyword(multi=True)


class SearchableDataset(IndexDocument):
    class Index:
        name = "dataset"

    title = Text(analyzer=dgv_analyzer)
    acronym = Text()
    url = Text()
    created_at = Date()
    last_update = Date()
    tags = Keyword(multi=True)
    license = Keyword()
    badges = Keyword(multi=True)
    frequency = Text()
    format = Keyword(multi=True)
    orga_sp = Integer()
    orga_followers = Float()
    views = Float()
    followers = Float()
    reuses = Float()
    featured = Integer()
    resources_count = Integer()
    resources_ids = Keyword(multi=True)
    resources_titles = Text(analyzer=dgv_analyzer)
    concat_title_org = Text(analyzer=dgv_analyzer)
    temporal_coverage_start = Date()
    temporal_coverage_end = Date()
    granularity = Keyword()
    geozones = Keyword(multi=True)
    description = Text(analyzer=dgv_analyzer)
    organization = Keyword()
    organization_name = Text(analyzer=dgv_analyzer, fields={"keyword": Keyword()})
    organization_with_id = Keyword()
    organization_badges = Keyword(multi=True)
    owner = Keyword()
    schema = Keyword(multi=True)
    topics = Keyword(multi=True)
    access_type = Keyword()
    format_family = Keyword(multi=True)
    producer_type = Keyword(multi=True)


ALL_DOCUMENT_CLASSES = [
    SearchableDataset,
    SearchableReuse,
    SearchableOrganization,
    SearchableDataservice,
    SearchableTopic,
    SearchableDiscussion,
    SearchablePost,
]


def configure_indices(prefix):
    for cls in ALL_DOCUMENT_CLASSES:
        if prefix:
            cls._index._name = f"{prefix}-{cls.Index.name}"
        else:
            cls._index._name = cls.Index.name


class ElasticClient:
    def __init__(self, url: str, prefix: str):
        self.es = connections.create_connection(hosts=[url])
        configure_indices(prefix)

    def init_indices(self) -> None:
        """
        Create templates based on Document mappings and map patterns.
        Create time-based index matchin the template patterns.
        """
        suffix_name = "-" + datetime.now(timezone.utc).strftime("%Y-%m-%d-%H-%M")

        for index_document in ALL_DOCUMENT_CLASSES:
            index_document.init_index(self.es, suffix_name)

    def delete_indices(self):
        """
        Clear all existing indices and templates.
        """
        for index_document in ALL_DOCUMENT_CLASSES:
            index_document.delete_indices(self.es)

    def delete_index(self, index_document: IndexDocument):
        index_document.delete_indices(self.es)

    def index_organization(self, to_index: Organization, index: str = None) -> None:
        SearchableOrganization(meta={"id": to_index.id}, **to_index.to_dict()).save(
            skip_empty=False, index=index
        )

    def index_dataset(self, to_index: Dataset, index: str = None) -> None:
        data = to_index.to_dict()
        if data.get("organization") and data.get("organization_name"):
            data["organization_with_id"] = f"{data['organization']}|{data['organization_name']}"
        SearchableDataset(meta={"id": to_index.id}, **data).save(skip_empty=False, index=index)

    def index_reuse(self, to_index: Reuse, index: str = None) -> None:
        data = to_index.to_dict()
        if data.get("organization") and data.get("organization_name"):
            data["organization_with_id"] = f"{data['organization']}|{data['organization_name']}"
        SearchableReuse(meta={"id": to_index.id}, **data).save(skip_empty=False, index=index)

    def index_dataservice(self, to_index: Dataservice, index: str = None) -> None:
        data = to_index.to_dict()
        if data.get("organization") and data.get("organization_name"):
            data["organization_with_id"] = f"{data['organization']}|{data['organization_name']}"
        SearchableDataservice(meta={"id": to_index.id}, **data).save(skip_empty=False, index=index)

    def index_topic(self, to_index: Topic, index: str = None) -> None:
        data = to_index.to_dict()
        if data.get("organization") and data.get("organization_name"):
            data["organization_with_id"] = f"{data['organization']}|{data['organization_name']}"
        SearchableTopic(meta={"id": to_index.id}, **data).save(skip_empty=False, index=index)

    def query_organizations(
        self,
        query_text: str,
        offset: int,
        page_size: int,
        filters: dict,
        sort: Optional[str] = None,
        facet_sizes: dict = {},
        facets: list = [],
    ) -> Tuple[int, List[dict], dict]:
        search = SearchableOrganization.search()

        post_filters = []
        for key, value in filters.items():
            post_filters.append(query.Q("term", **{key: value}))

        organizations_score_functions = [
            query.SF("field_value_factor", field="orga_sp", factor=8, modifier="sqrt", missing=1),
            query.SF("field_value_factor", field="followers", factor=4, modifier="sqrt", missing=1),
            query.SF("field_value_factor", field="views", factor=1, modifier="sqrt", missing=1),
        ]

        if query_text:
            search = search.query(
                "bool",
                should=[
                    query.Q(
                        "function_score",
                        query=query.Bool(
                            should=[
                                query.MultiMatch(
                                    query=query_text,
                                    type="phrase",
                                    fields=["id^15", "name^15", "acronym^15", "description^8"],
                                )
                            ]
                        ),
                        functions=organizations_score_functions,
                    ),
                    query.Q(
                        "function_score",
                        query=query.Bool(
                            should=[
                                query.MultiMatch(
                                    query=query_text,
                                    type="cross_fields",
                                    fields=["id^15", "name^7", "acronym^7", "description^4"],
                                    operator="and",
                                )
                            ]
                        ),
                        functions=organizations_score_functions,
                    ),
                    query.Match(title={"query": query_text, "fuzziness": "AUTO:4,6"}),
                ],
            )
        else:
            search = search.query(
                query.Q(
                    "function_score",
                    query=query.MatchAll(),
                    functions=organizations_score_functions,
                )
            )

        for facet in facets:
            if isinstance(facet, TermsFacet):
                search.aggs.bucket(
                    facet.name, "terms", field=facet.es_field, size=facet_sizes.get(facet.name, 50)
                )
        search.aggs.metric("total_count", "cardinality", field="_id")

        if post_filters:
            search = search.post_filter(query.Bool(must=post_filters))

        if sort:
            search = search.sort(sort, {"_score": {"order": "desc"}})

        search = search[offset : (offset + page_size)]

        response = search.execute()
        results_number = response.hits.total.value
        if response.hits and not isinstance(response.hits[0], SearchableOrganization):
            raise ValueError(
                "Results are not of SearchableOrganization type. It probably means that index analyzers "
                "were not correctly set using template patterns on index initialization."
            )
        res = [hit.to_dict(skip_empty=False) for hit in response.hits]

        facets_result = {}
        if hasattr(response, "aggregations"):
            total_count = (
                int(response.aggregations.total_count.value)
                if hasattr(response.aggregations, "total_count")
                else 0
            )
            for facet in facets:
                if isinstance(facet, TermsFacet) and hasattr(response.aggregations, facet.name):
                    buckets = [
                        {"name": b.key, "count": b.doc_count}
                        for b in response.aggregations[facet.name].buckets
                    ]
                    facets_result[facet.name] = [{"name": "all", "count": total_count}] + buckets

        return results_number, res, facets_result

    def query_topics(
        self,
        query_text: str,
        offset: int,
        page_size: int,
        filters: dict,
        sort: Optional[str] = None,
        facet_sizes: dict = {},
        facets: list = [],
    ) -> Tuple[int, List[dict], dict]:
        search = SearchableTopic.search()

        last_update_range_mapping = {
            "last_30_days": "now-30d/d",
            "last_12_months": "now-12M/d",
            "last_3_years": "now-3y/d",
        }

        # Build filters dictionary by category
        filter_dict = {
            "organization_id_with_name": None,
            "producer_type": None,
            "last_update_range": None,
            "other": [],
        }

        for key, value in filters.items():
            if key == "last_update_range":
                if value in last_update_range_mapping:
                    filter_dict["last_update_range"] = query.Q(
                        "range", last_modified={"gte": last_update_range_mapping[value]}
                    )
            elif key == "tag":
                if isinstance(value, list):
                    tag_filters = [query.Q("term", tags=tag) for tag in value]
                    filter_dict["other"].append(query.Bool(must=tag_filters))
                else:
                    filter_dict["other"].append(query.Q("term", tags=value))
            elif key == "organization" and isinstance(value, list):
                list_filters = [query.Q("term", organization=v) for v in value]
                filter_dict["organization_id_with_name"] = query.Bool(
                    should=list_filters, minimum_should_match=1
                )
            elif key == "organization_id_with_name" and isinstance(value, list):
                list_filters = [query.Q("term", organization_with_id=v) for v in value]
                filter_dict["organization_id_with_name"] = query.Bool(
                    should=list_filters, minimum_should_match=1
                )
            elif key == "organization_id_with_name":
                filter_dict["organization_id_with_name"] = query.Q(
                    "term", organization_with_id=value
                )
            elif key == "producer_type":
                filter_dict["producer_type"] = query.Q("term", producer_type=value)
            else:
                filter_dict["other"].append(query.Q("term", **{key: value}))

        if query_text:
            search = search.query(
                "bool",
                should=[
                    query.MultiMatch(
                        query=query_text,
                        type="most_fields",
                        operator="and",
                        fields=["id^5", "name^10", "description^4", "tags^3"],
                        fuzziness="AUTO:4,6",
                    )
                ],
            )
        else:
            search = search.query(query.MatchAll())

        def get_filters_except(exclude_key):
            filters_list = filter_dict["other"].copy()
            for key in ["organization_id_with_name", "producer_type", "last_update_range"]:
                if key != exclude_key and filter_dict[key] is not None:
                    filters_list.append(filter_dict[key])
            return filters_list

        for facet in facets:
            if isinstance(facet, TermsFacet):
                size = facet_sizes.get(facet.name, 50)
                f = get_filters_except(facet.name)
                if f:
                    agg = search.aggs.bucket(
                        f"{facet.name}_filtered", "filter", filter=query.Bool(must=f)
                    )
                    agg.bucket(facet.name, "terms", field=facet.es_field, size=size)
                    agg.metric("total", "cardinality", field="_id")
                else:
                    search.aggs.bucket(facet.name, "terms", field=facet.es_field, size=size)
                    search.aggs.metric(f"{facet.name}_total", "cardinality", field="_id")
            elif isinstance(facet, DateRangeFacet):
                f = get_filters_except("last_update_range")
                if f:
                    agg = search.aggs.bucket(
                        "last_update_filtered", "filter", filter=query.Bool(must=f)
                    )
                    agg.bucket(
                        "last_update", "date_range", field=facet.es_field, ranges=DATE_RANGES
                    )
                    agg.metric("total", "cardinality", field="_id")
                else:
                    search.aggs.bucket(
                        "last_update", "date_range", field=facet.es_field, ranges=DATE_RANGES
                    )
                    search.aggs.metric("last_update_total", "cardinality", field="_id")

        post_filters = []
        for key, value in filter_dict.items():
            if key != "other" and value is not None:
                post_filters.append(value)
        post_filters.extend(filter_dict["other"])

        if post_filters:
            search = search.post_filter(query.Bool(must=post_filters))

        if sort:
            sort_field = sort
            if sort == "last_update":
                sort_field = "last_modified"
            elif sort == "-last_update":
                sort_field = "-last_modified"
            elif sort == "name":
                sort_field = "name.keyword"
            elif sort == "-name":
                sort_field = "-name.keyword"
            search = search.sort(sort_field, {"_score": {"order": "desc"}})

        search = search[offset : (offset + page_size)]

        response = search.execute()
        results_number = response.hits.total.value
        if response.hits and not isinstance(response.hits[0], SearchableTopic):
            raise ValueError(
                "Results are not of SearchableTopic type. It probably means that index analyzers were not correctly set "
                "using template patterns on index initialization."
            )
        res = [hit.to_dict(skip_empty=False) for hit in response.hits]

        facets_result = {}
        if hasattr(response, "aggregations"):
            facets_result = _parse_filtered_facets(response.aggregations, facets)

        return results_number, res, facets_result

    def query_datasets(
        self,
        query_text: str,
        offset: int,
        page_size: int,
        filters: dict,
        sort: Optional[str] = None,
        facet_sizes: dict = {},
        facets: list = [],
    ) -> Tuple[int, List[dict], dict]:
        search = SearchableDataset.search()

        last_update_range_mapping = {
            "last_30_days": "now-30d/d",
            "last_12_months": "now-12M/d",
            "last_3_years": "now-3y/d",
        }

        filter_dict = {
            "format_family": None,
            "access_type": None,
            "producer_type": None,
            "organization_id_with_name": None,
            "last_update_range": None,
            "tag": None,
            "license": None,
            "format": None,
            "schema": None,
            "geozone": None,
            "granularity": None,
            "badge": None,
            "topics": None,
            "other": [],
        }

        for key, value in filters.items():
            if key == "temporal_coverage_start":
                filter_dict["other"].append(
                    query.Q("range", temporal_coverage_start={"lte": value})
                )
            elif key == "temporal_coverage_end":
                filter_dict["other"].append(query.Q("range", temporal_coverage_end={"gte": value}))
            elif key == "last_update_range":
                if value in last_update_range_mapping:
                    filter_dict["last_update_range"] = query.Q(
                        "range", last_update={"gte": last_update_range_mapping[value]}
                    )
            elif key == "tags":
                tag_filters = [query.Q("term", tags=tag) for tag in value]
                filter_dict["other"].append(query.Bool(must=tag_filters))
            elif key in ["license", "format", "schema", "geozones", "granularity", "badges"]:
                filter_key = {"geozones": "geozone", "badges": "badge"}.get(key, key)
                if isinstance(value, list):
                    list_filters = [query.Q("term", **{key: v}) for v in value]
                    filter_dict[filter_key] = query.Bool(
                        should=list_filters, minimum_should_match=1
                    )
                else:
                    filter_dict[filter_key] = query.Q("term", **{key: value})
            elif key == "topics":
                if isinstance(value, list):
                    topic_filters = [query.Q("term", topics=topic) for topic in value]
                    filter_dict["topics"] = query.Bool(should=topic_filters, minimum_should_match=1)
                else:
                    filter_dict["topics"] = query.Q("term", topics=value)
            elif key == "organization_id_with_name":
                if isinstance(value, list):
                    list_filters = [query.Q("term", organization=v) for v in value]
                    filter_dict[key] = query.Bool(should=list_filters, minimum_should_match=1)
                else:
                    filter_dict[key] = query.Q("term", **{"organization": value})
            elif key in ["format_family", "access_type", "producer_type", "tag"]:
                filter_dict[key] = query.Q("term", **{key: value})
            else:
                filter_dict["other"].append(query.Q("term", **{key: value}))

        datasets_score_functions = [
            query.SF("field_value_factor", field="orga_sp", factor=8, modifier="sqrt", missing=1),
            query.SF("field_value_factor", field="views", factor=4, modifier="sqrt", missing=1),
            query.SF("field_value_factor", field="followers", factor=4, modifier="sqrt", missing=1),
            query.SF(
                "field_value_factor", field="orga_followers", factor=1, modifier="sqrt", missing=1
            ),
            query.SF("field_value_factor", field="featured", factor=1, modifier="sqrt", missing=1),
        ]

        if query_text:
            search = search.query(
                "bool",
                should=[
                    query.Q(
                        "function_score",
                        query=query.Bool(
                            should=[
                                query.MultiMatch(
                                    query=query_text,
                                    type="phrase",
                                    fields=[
                                        "id^15",
                                        "title^15",
                                        "acronym^15",
                                        "description^8",
                                        "organization_name^8",
                                        "resources_ids^8",
                                        "resources_titles^5",
                                    ],
                                )
                            ]
                        ),
                        functions=datasets_score_functions,
                    ),
                    query.Q(
                        "function_score",
                        query=query.Bool(
                            must=[
                                query.Match(
                                    concat_title_org={
                                        "query": query_text,
                                        "operator": "and",
                                        "boost": 8,
                                    }
                                )
                            ]
                        ),
                        functions=datasets_score_functions,
                    ),
                    query.Q(
                        "function_score",
                        query=query.Bool(
                            should=[
                                query.MultiMatch(
                                    query=query_text,
                                    type="cross_fields",
                                    fields=[
                                        "id^7",
                                        "title^7",
                                        "acronym^7",
                                        "description^4",
                                        "organization_name^4",
                                        "resources_ids^4",
                                        "resources_titles^2",
                                    ],
                                    operator="and",
                                )
                            ]
                        ),
                        functions=datasets_score_functions,
                    ),
                    query.MultiMatch(
                        query=query_text,
                        type="most_fields",
                        operator="and",
                        fields=["title", "organization_name"],
                        fuzziness="AUTO:4,6",
                    ),
                ],
            )
        else:
            search = search.query(
                query.Q(
                    "function_score", query=query.MatchAll(), functions=datasets_score_functions
                )
            )

        def get_filters_except(exclude_key):
            filters_list = filter_dict["other"].copy()
            for key in [
                "format_family",
                "access_type",
                "producer_type",
                "organization_id_with_name",
                "last_update_range",
                "tag",
                "license",
                "format",
                "schema",
                "geozone",
                "granularity",
                "topics",
            ]:
                if key != exclude_key and filter_dict[key] is not None:
                    filters_list.append(filter_dict[key])
            return filters_list

        for facet in facets:
            if isinstance(facet, TermsFacet):
                size = facet_sizes.get(facet.name, 50)
                f = get_filters_except(facet.name)
                if f:
                    agg = search.aggs.bucket(
                        f"{facet.name}_filtered", "filter", filter=query.Bool(must=f)
                    )
                    agg.bucket(facet.name, "terms", field=facet.es_field, size=size)
                    agg.metric("total", "cardinality", field="_id")
                else:
                    search.aggs.bucket(facet.name, "terms", field=facet.es_field, size=size)
                    search.aggs.metric(f"{facet.name}_total", "cardinality", field="_id")
            elif isinstance(facet, DateRangeFacet):
                f = get_filters_except("last_update_range")
                if f:
                    agg = search.aggs.bucket(
                        "last_update_filtered", "filter", filter=query.Bool(must=f)
                    )
                    agg.bucket(
                        "last_update", "date_range", field=facet.es_field, ranges=DATE_RANGES
                    )
                    agg.metric("total", "cardinality", field="_id")
                else:
                    search.aggs.bucket(
                        "last_update", "date_range", field=facet.es_field, ranges=DATE_RANGES
                    )
                    search.aggs.metric("last_update_total", "cardinality", field="_id")

        post_filters = []
        for key, value in filter_dict.items():
            if key != "other" and value is not None:
                post_filters.append(value)
        post_filters.extend(filter_dict["other"])

        if post_filters:
            search = search.post_filter(query.Bool(must=post_filters))

        if sort:
            search = search.sort(sort, {"_score": {"order": "desc"}})

        search = search[offset : (offset + page_size)]

        log.debug("Elasticsearch query for datasets: %s", search.to_dict())
        response = search.execute()
        results_number = response.hits.total.value
        if response.hits and not isinstance(response.hits[0], SearchableDataset):
            raise ValueError(
                "Results are not of SearchableDataset type. It probably means that index analyzers were not correctly set "
                "using template patterns on index initialization."
            )
        res = [hit.to_dict(skip_empty=False) for hit in response.hits]

        facets_result = {}
        if hasattr(response, "aggregations"):
            facets_result = _parse_filtered_facets(response.aggregations, facets)

        return results_number, res, facets_result

    def query_reuses(
        self,
        query_text: str,
        offset: int,
        page_size: int,
        filters: dict,
        sort: Optional[str] = None,
        facet_sizes: dict = {},
        facets: list = [],
    ) -> Tuple[int, List[dict], dict]:
        search = SearchableReuse.search()

        last_update_range_mapping = {
            "last_30_days": "now-30d/d",
            "last_12_months": "now-12M/d",
            "last_3_years": "now-3y/d",
        }

        filter_dict = {
            "producer_type": None,
            "organization_id_with_name": None,
            "topic_object": None,
            "type": None,
            "topic": None,
            "tag": None,
            "badge": None,
            "last_update_range": None,
            "other": [],
        }

        # ---- build filters
        for key, value in filters.items():
            if key == "last_update_range" and value in last_update_range_mapping:
                filter_dict["last_update_range"] = query.Q(
                    "range", last_modified={"gte": last_update_range_mapping[value]}
                )

            elif key == "tags":
                if isinstance(value, list):
                    tag_filters = [query.Q("term", tags=v) for v in value]
                    filter_dict["other"].append(query.Bool(must=tag_filters))
                else:
                    filter_dict["tag"] = query.Q("term", tags=value)

            elif key == "tag":
                filter_dict["tag"] = query.Q("term", tags=value)

            elif key == "badges":
                if isinstance(value, list):
                    badge_filters = [query.Q("term", badges=v) for v in value]
                    filter_dict["badge"] = query.Bool(should=badge_filters, minimum_should_match=1)
                else:
                    filter_dict["badge"] = query.Q("term", badges=value)

            elif key == "badge":
                filter_dict["badge"] = query.Q("term", badges=value)

            elif key == "topic_object":
                if isinstance(value, list):
                    topic_filters = [query.Q("term", topic_object=v) for v in value]
                    filter_dict["topic_object"] = query.Bool(
                        should=topic_filters, minimum_should_match=1
                    )
                else:
                    filter_dict["topic_object"] = query.Q("term", topic_object=value)

            elif key == "organization_id_with_name":
                if isinstance(value, list):
                    org_filters = [query.Q("term", organization=v) for v in value]
                    filter_dict[key] = query.Bool(should=org_filters, minimum_should_match=1)
                else:
                    filter_dict[key] = query.Q("term", organization=value)

            elif key == "organization":
                if isinstance(value, list):
                    org_filters = [query.Q("term", organization=v) for v in value]
                    filter_dict["other"].append(
                        query.Bool(should=org_filters, minimum_should_match=1)
                    )
                else:
                    filter_dict["other"].append(query.Q("term", organization=value))

            elif key in ["producer_type", "type", "topic"]:
                filter_dict[key] = query.Q("term", **{key: value})

            else:
                filter_dict["other"].append(query.Q("term", **{key: value}))

        reuses_score_functions = [
            query.SF("field_value_factor", field="views", factor=4, modifier="sqrt", missing=1),
            query.SF("field_value_factor", field="followers", factor=4, modifier="sqrt", missing=1),
            query.SF(
                "field_value_factor", field="orga_followers", factor=1, modifier="sqrt", missing=1
            ),
            query.SF("field_value_factor", field="featured", factor=1, modifier="sqrt", missing=1),
            query.SF("script_score", script={"source": "doc['archived'].size() == 0 ? 1 : 0.2"}),
        ]

        if query_text:
            search = search.query(
                "bool",
                should=[
                    query.Q(
                        "function_score",
                        query=query.Bool(
                            should=[
                                query.MultiMatch(
                                    query=query_text,
                                    type="phrase",
                                    fields=[
                                        "id^15",
                                        "title^15",
                                        "description^8",
                                        "organization_name^8",
                                    ],
                                )
                            ]
                        ),
                        functions=reuses_score_functions,
                    ),
                    query.Q(
                        "function_score",
                        query=query.Bool(
                            should=[
                                query.MultiMatch(
                                    query=query_text,
                                    type="cross_fields",
                                    fields=[
                                        "id^7",
                                        "title^7",
                                        "description^4",
                                        "organization_name^4",
                                    ],
                                    operator="and",
                                )
                            ]
                        ),
                        functions=reuses_score_functions,
                    ),
                    query.MultiMatch(
                        query=query_text,
                        type="most_fields",
                        operator="and",
                        fields=["title", "organization_name"],
                        fuzziness="AUTO:4,6",
                    ),
                ],
            )
        else:
            search = search.query(
                query.Q("function_score", query=query.MatchAll(), functions=reuses_score_functions)
            )

        def get_filters_except(exclude_key: str):
            flt = list(filter_dict["other"])
            for k in [
                "producer_type",
                "organization_id_with_name",
                "topic_object",
                "type",
                "topic",
                "tag",
                "badge",
                "last_update_range",
            ]:
                if k != exclude_key and filter_dict[k] is not None:
                    flt.append(filter_dict[k])
            return flt

        for facet in facets:
            if isinstance(facet, TermsFacet):
                size = facet_sizes.get(facet.name, 50)
                f = get_filters_except(facet.name)
                if f:
                    agg = search.aggs.bucket(
                        f"{facet.name}_filtered", "filter", filter=query.Bool(must=f)
                    )
                    agg.bucket(facet.name, "terms", field=facet.es_field, size=size)
                    agg.metric("total", "cardinality", field="_id")
                else:
                    search.aggs.bucket(facet.name, "terms", field=facet.es_field, size=size)
                    search.aggs.metric(f"{facet.name}_total", "cardinality", field="_id")
            elif isinstance(facet, DateRangeFacet):
                f = get_filters_except("last_update_range")
                if f:
                    agg = search.aggs.bucket(
                        "last_update_filtered", "filter", filter=query.Bool(must=f)
                    )
                    agg.bucket(
                        "last_update", "date_range", field=facet.es_field, ranges=DATE_RANGES
                    )
                    agg.metric("total", "cardinality", field="_id")
                else:
                    search.aggs.bucket(
                        "last_update", "date_range", field=facet.es_field, ranges=DATE_RANGES
                    )
                    search.aggs.metric("last_update_total", "cardinality", field="_id")

        post_filters = []
        for k in [
            "producer_type",
            "organization_id_with_name",
            "topic_object",
            "type",
            "topic",
            "tag",
            "badge",
            "last_update_range",
        ]:
            if filter_dict[k] is not None:
                post_filters.append(filter_dict[k])
        post_filters.extend(filter_dict["other"])
        if post_filters:
            search = search.post_filter(query.Bool(must=post_filters))

        if sort:
            search = search.sort(sort, {"_score": {"order": "desc"}})

        search = search[offset : (offset + page_size)]
        response = search.execute()

        results_number = response.hits.total.value
        if response.hits and not isinstance(response.hits[0], SearchableReuse):
            raise ValueError(
                "Results are not of SearchableReuse type. It probably means that index analyzers were not correctly set "
                "using template patterns on index initialization."
            )

        res = [hit.to_dict(skip_empty=False) for hit in response.hits]

        facets_result = {}
        if hasattr(response, "aggregations"):
            facets_result = _parse_filtered_facets(response.aggregations, facets)

        return results_number, res, facets_result

    def query_dataservices(
        self,
        query_text: str,
        offset: int,
        page_size: int,
        filters: dict,
        sort: Optional[str] = None,
        facet_sizes: dict = {},
        facets: list = [],
    ):
        search = SearchableDataservice.search()

        last_update_range_mapping = {
            "last_30_days": "now-30d/d",
            "last_12_months": "now-12M/d",
            "last_3_years": "now-3y/d",
        }

        filter_dict = {
            "access_type": None,
            "producer_type": None,
            "organization_id_with_name": None,
            "topics": None,
            "tag": None,
            "badge": None,
            "last_update_range": None,
            "other": [],
        }

        for key, value in filters.items():
            if key == "last_update_range" and value in last_update_range_mapping:
                filter_dict["last_update_range"] = query.Q(
                    "range", metadata_modified_at={"gte": last_update_range_mapping[value]}
                )

            elif key == "tags":
                if isinstance(value, list):
                    tag_filters = [query.Q("term", tags=v) for v in value]
                    filter_dict["other"].append(query.Bool(must=tag_filters))
                else:
                    filter_dict["tag"] = query.Q("term", tags=value)

            elif key == "tag":
                filter_dict["tag"] = query.Q("term", tags=value)

            elif key == "topics":
                if isinstance(value, list):
                    topic_filters = [query.Q("term", topics=v) for v in value]
                    filter_dict["topics"] = query.Bool(should=topic_filters, minimum_should_match=1)
                else:
                    filter_dict["topics"] = query.Q("term", topics=value)

            elif key == "organization_id_with_name":
                if isinstance(value, list):
                    org_filters = [query.Q("term", organization=v) for v in value]
                    filter_dict[key] = query.Bool(should=org_filters, minimum_should_match=1)
                else:
                    filter_dict[key] = query.Q("term", organization=value)

            elif key == "producer_type":
                filter_dict["producer_type"] = query.Q("term", producer_type=value)

            elif key == "access_type":
                filter_dict["access_type"] = query.Q("term", access_type=value)

            elif key in ("badge", "badges"):
                if isinstance(value, list):
                    badge_filters = [query.Q("term", badges=v) for v in value]
                    filter_dict["badge"] = query.Bool(should=badge_filters, minimum_should_match=1)
                else:
                    filter_dict["badge"] = query.Q("term", badges=value)

            else:
                filter_dict["other"].append(query.Q("term", **{key: value}))

        dataservices_score_functions = [
            query.SF(
                "field_value_factor",
                field="description_length",
                factor=1,
                modifier="sqrt",
                missing=1,
            ),
            query.SF("field_value_factor", field="views", factor=4, modifier="sqrt", missing=1),
            query.SF("field_value_factor", field="followers", factor=4, modifier="sqrt", missing=1),
            query.SF(
                "field_value_factor", field="orga_followers", factor=1, modifier="sqrt", missing=1
            ),
        ]

        if query_text:
            search = search.query(
                "bool",
                should=[
                    query.Q(
                        "function_score",
                        query=query.Bool(
                            should=[
                                query.MultiMatch(
                                    query=query_text,
                                    type="phrase",
                                    fields=[
                                        "id^15",
                                        "title^15",
                                        "description^8",
                                        "organization_name^8",
                                        "documentation_content^3",
                                    ],
                                )
                            ]
                        ),
                        functions=dataservices_score_functions,
                    ),
                    query.Q(
                        "function_score",
                        query=query.Bool(
                            should=[
                                query.MultiMatch(
                                    query=query_text,
                                    type="cross_fields",
                                    fields=[
                                        "id^7",
                                        "title^7",
                                        "description^4",
                                        "organization_name^4",
                                        "documentation_content^2",
                                    ],
                                    operator="and",
                                )
                            ]
                        ),
                        functions=dataservices_score_functions,
                    ),
                    query.MultiMatch(
                        query=query_text,
                        type="most_fields",
                        operator="and",
                        fields=["title", "organization_name", "documentation_content"],
                        fuzziness="AUTO:4,6",
                    ),
                ],
            )
        else:
            search = search.query(
                query.Q(
                    "function_score", query=query.MatchAll(), functions=dataservices_score_functions
                )
            )

        def get_filters_except(exclude_key: str):
            filters_list = list(filter_dict["other"])
            for k in [
                "access_type",
                "producer_type",
                "organization_id_with_name",
                "topics",
                "tag",
                "badge",
                "last_update_range",
            ]:
                if k != exclude_key and filter_dict[k] is not None:
                    filters_list.append(filter_dict[k])
            return filters_list

        for facet in facets:
            if isinstance(facet, TermsFacet):
                size = facet_sizes.get(facet.name, 50)
                f = get_filters_except(facet.name)
                if f:
                    agg = search.aggs.bucket(
                        f"{facet.name}_filtered", "filter", filter=query.Bool(must=f)
                    )
                    agg.bucket(facet.name, "terms", field=facet.es_field, size=size)
                    agg.metric("total", "cardinality", field="_id")
                else:
                    search.aggs.bucket(facet.name, "terms", field=facet.es_field, size=size)
                    search.aggs.metric(f"{facet.name}_total", "cardinality", field="_id")
            elif isinstance(facet, DateRangeFacet):
                f = get_filters_except("last_update_range")
                if f:
                    agg = search.aggs.bucket(
                        "last_update_filtered", "filter", filter=query.Bool(must=f)
                    )
                    agg.bucket(
                        "last_update", "date_range", field=facet.es_field, ranges=DATE_RANGES
                    )
                    agg.metric("total", "cardinality", field="_id")
                else:
                    search.aggs.bucket(
                        "last_update", "date_range", field=facet.es_field, ranges=DATE_RANGES
                    )
                    search.aggs.metric("last_update_total", "cardinality", field="_id")

        post_filters = []
        for k in [
            "access_type",
            "producer_type",
            "organization_id_with_name",
            "topics",
            "tag",
            "badge",
            "last_update_range",
        ]:
            if filter_dict[k] is not None:
                post_filters.append(filter_dict[k])
        post_filters.extend(filter_dict["other"])
        if post_filters:
            search = search.post_filter(query.Bool(must=post_filters))

        if sort:
            search = search.sort(sort, {"_score": {"order": "desc"}})

        search = search[offset : (offset + page_size)]
        response = search.execute()

        results_number = response.hits.total.value
        res = [hit.to_dict(skip_empty=False) for hit in response.hits]

        facets_result = {}
        if hasattr(response, "aggregations"):
            facets_result = _parse_filtered_facets(response.aggregations, facets)

        return results_number, res, facets_result

    def find_one_organization(self, organization_id: str) -> Optional[dict]:
        try:
            return SearchableOrganization.get(id=organization_id).to_dict()
        except NotFoundError:
            return None

    def find_one_dataset(self, dataset_id: str) -> Optional[dict]:
        try:
            return SearchableDataset.get(id=dataset_id).to_dict()
        except NotFoundError:
            return None

    def find_one_reuse(self, reuse_id: str) -> Optional[dict]:
        try:
            return SearchableReuse.get(id=reuse_id).to_dict()
        except NotFoundError:
            return None

    def find_one_dataservice(self, dataservice_id: str) -> Optional[dict]:
        try:
            return SearchableDataservice.get(id=dataservice_id).to_dict()
        except NotFoundError:
            return None

    def find_one_topic(self, topic_id: str) -> Optional[dict]:
        try:
            return SearchableTopic.get(id=topic_id).to_dict()
        except NotFoundError:
            return None

    def delete_one_organization(self, organization_id: str) -> Optional[str]:
        try:
            SearchableOrganization.get(id=organization_id).delete()
            return organization_id
        except NotFoundError:
            return None

    def delete_one_dataset(self, dataset_id: str) -> Optional[str]:
        try:
            SearchableDataset.get(id=dataset_id).delete()
            return dataset_id
        except NotFoundError:
            return None

    def delete_one_reuse(self, reuse_id: str) -> Optional[str]:
        try:
            SearchableReuse.get(id=reuse_id).delete()
            return reuse_id
        except NotFoundError:
            return None

    def delete_one_dataservice(self, dataservice_id: str) -> Optional[str]:
        try:
            SearchableDataservice.get(id=dataservice_id).delete()
            return dataservice_id
        except NotFoundError:
            return None

    def delete_one_topic(self, topic_id: str) -> Optional[str]:
        try:
            SearchableTopic.get(id=topic_id).delete()
            return topic_id
        except NotFoundError:
            return None

    def index_discussion(self, to_index: Discussion, index: str = None) -> None:
        SearchableDiscussion(meta={"id": to_index.id}, **to_index.to_dict()).save(
            skip_empty=False, index=index
        )

    def query_discussions(
        self,
        query_text: str,
        offset: int,
        page_size: int,
        filters: dict,
        sort: Optional[str] = None,
        facet_sizes: dict = {},
        facets: list = [],
    ) -> Tuple[int, List[dict], dict]:
        search = SearchableDiscussion.search()

        last_update_range_mapping = {
            "last_30_days": "now-30d/d",
            "last_12_months": "now-12M/d",
            "last_3_years": "now-3y/d",
        }

        post_filters = []

        for key, value in filters.items():
            if key == "last_update_range":
                if value in last_update_range_mapping:
                    post_filters.append(
                        query.Q("range", created_at={"gte": last_update_range_mapping[value]})
                    )
            elif key == "object_type":
                post_filters.append(query.Q("term", subject_class=value))
            else:
                post_filters.append(query.Q("term", **{key: value}))

        if query_text:
            search = search.query(
                "bool",
                should=[
                    query.MultiMatch(
                        query=query_text,
                        type="most_fields",
                        operator="and",
                        fields=["id^5", "title^10", "content^4"],
                        fuzziness="AUTO:4,6",
                    )
                ],
            )
        else:
            search = search.query(query.MatchAll())

        for facet in facets:
            if isinstance(facet, TermsFacet):
                search.aggs.bucket(
                    facet.name, "terms", field=facet.es_field, size=facet_sizes.get(facet.name, 50)
                )
            elif isinstance(facet, DateRangeFacet):
                search.aggs.bucket(
                    "last_update", "date_range", field=facet.es_field, ranges=DATE_RANGES
                )
        search.aggs.metric("total_count", "cardinality", field="_id")

        if post_filters:
            search = search.post_filter(query.Bool(must=post_filters))

        if sort:
            search = search.sort(sort, {"_score": {"order": "desc"}})

        search = search[offset : (offset + page_size)]

        response = search.execute()
        results_number = response.hits.total.value
        if response.hits and not isinstance(response.hits[0], SearchableDiscussion):
            raise ValueError(
                "Results are not of SearchableDiscussion type. It probably means that index analyzers were not correctly set "
                "using template patterns on index initialization."
            )
        res = [hit.to_dict(skip_empty=False) for hit in response.hits]

        facets_result = {}
        if hasattr(response, "aggregations"):
            total_count = (
                int(response.aggregations.total_count.value)
                if hasattr(response.aggregations, "total_count")
                else 0
            )
            for facet in facets:
                agg_name = facet.name if isinstance(facet, TermsFacet) else "last_update"
                if hasattr(response.aggregations, agg_name):
                    buckets = [
                        {"name": b.key, "count": b.doc_count}
                        for b in response.aggregations[agg_name].buckets
                    ]
                    facets_result[agg_name] = [{"name": "all", "count": total_count}] + buckets

        return results_number, res, facets_result

    def find_one_discussion(self, discussion_id: str) -> Optional[dict]:
        try:
            return SearchableDiscussion.get(id=discussion_id).to_dict()
        except NotFoundError:
            return None

    def delete_one_discussion(self, discussion_id: str) -> Optional[str]:
        try:
            SearchableDiscussion.get(id=discussion_id).delete()
            return discussion_id
        except NotFoundError:
            return None

    def index_post(self, to_index: Post, index: str = None) -> None:
        SearchablePost(meta={"id": to_index.id}, **to_index.to_dict()).save(
            skip_empty=False, index=index
        )

    def query_posts(
        self,
        query_text: str,
        offset: int,
        page_size: int,
        filters: dict,
        sort: Optional[str] = None,
        facet_sizes: dict = {},
        facets: list = [],
    ) -> Tuple[int, List[dict], dict]:
        search = SearchablePost.search()

        last_modified_range_mapping = {
            "last_30_days": "now-30d/d",
            "last_12_months": "now-12M/d",
            "last_3_years": "now-3y/d",
        }

        post_filters = []

        for key, value in filters.items():
            if key == "last_update_range":
                if value in last_modified_range_mapping:
                    post_filters.append(
                        query.Q("range", last_modified={"gte": last_modified_range_mapping[value]})
                    )
            elif key == "tags":
                if isinstance(value, list):
                    tag_filters = [query.Q("term", tags=tag) for tag in value]
                    post_filters.append(query.Bool(must=tag_filters))
                else:
                    post_filters.append(query.Q("term", tags=value))
            else:
                post_filters.append(query.Q("term", **{key: value}))

        if query_text:
            search = search.query(
                "bool",
                should=[
                    query.MultiMatch(
                        query=query_text,
                        type="most_fields",
                        operator="and",
                        fields=["id^5", "name^10", "headline^7", "content^4", "tags^3"],
                        fuzziness="AUTO:4,6",
                    )
                ],
            )
        else:
            search = search.query(query.MatchAll())

        for facet in facets:
            if isinstance(facet, DateRangeFacet):
                search.aggs.bucket(
                    "last_update", "date_range", field=facet.es_field, ranges=DATE_RANGES
                )
        search.aggs.metric("total_count", "cardinality", field="_id")

        if post_filters:
            search = search.post_filter(query.Bool(must=post_filters))

        if sort:
            search = search.sort(sort, {"_score": {"order": "desc"}})

        search = search[offset : (offset + page_size)]

        response = search.execute()
        results_number = response.hits.total.value
        if response.hits and not isinstance(response.hits[0], SearchablePost):
            raise ValueError(
                "Results are not of SearchablePost type. It probably means that index analyzers were not correctly set "
                "using template patterns on index initialization."
            )
        res = [hit.to_dict(skip_empty=False) for hit in response.hits]

        facets_result = {}
        if hasattr(response, "aggregations"):
            total_count = (
                int(response.aggregations.total_count.value)
                if hasattr(response.aggregations, "total_count")
                else 0
            )
            if hasattr(response.aggregations, "last_update"):
                buckets = [
                    {"name": b.key, "count": b.doc_count}
                    for b in response.aggregations.last_update.buckets
                ]
                facets_result["last_update"] = [{"name": "all", "count": total_count}] + buckets

        return results_number, res, facets_result

    def find_one_post(self, post_id: str) -> Optional[dict]:
        try:
            return SearchablePost.get(id=post_id).to_dict()
        except NotFoundError:
            return None

    def delete_one_post(self, post_id: str) -> Optional[str]:
        try:
            SearchablePost.get(id=post_id).delete()
            return post_id
        except NotFoundError:
            return None
