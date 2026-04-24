from math import ceil
from typing import List, Optional, Tuple

from udata_search_service.entities import (
    Dataservice,
    Dataset,
    Discussion,
    EntityBase,
    Organization,
    Post,
    Reuse,
    Topic,
)
from udata_search_service.search_clients import (
    DateRangeFacet,
    ElasticClient,
    TermsFacet,
)


class BaseService:
    entity_class: type[EntityBase]
    entity_name: str
    facets: list = []

    # {filter_param_name: elasticsearch_field_name}
    filter_renames: dict[str, str] = {}
    # {sort_param_name: elasticsearch_field_name}
    sort_renames: dict[str, str] = {"created": "created_at"}

    def __init__(self, search_client: ElasticClient):
        self.search_client = search_client
        self._client_index = getattr(search_client, f"index_{self.entity_name}")
        self._client_query = getattr(search_client, f"query_{self.entity_name}s")
        self._client_find_one = getattr(search_client, f"find_one_{self.entity_name}")
        self._client_delete_one = getattr(search_client, f"delete_one_{self.entity_name}")

    def feed(self, entity: EntityBase, index: str = None) -> None:
        self._client_index(entity, index)

    def search(self, filters: dict) -> Tuple[List[EntityBase], int, int, dict]:
        page = filters.pop("page")
        page_size = filters.pop("page_size")
        search_text = filters.pop("q")
        sort = self.format_sort(filters.pop("sort", None))
        facet_sizes = filters.pop("facet_sizes", {})

        offset = page_size * (page - 1) if page > 1 else 0

        self.format_filters(filters)

        results_number, search_results, facets = self._client_query(
            search_text,
            offset,
            page_size,
            filters,
            sort,
            facet_sizes=facet_sizes,
            facets=self.__class__.facets,
        )
        results = [self.entity_class.load_from_dict(hit) for hit in search_results]
        total_pages = ceil(results_number / page_size) or 1
        return results, results_number, total_pages, facets

    def find_one(self, entity_id: str) -> Optional[EntityBase]:
        try:
            return self.entity_class.load_from_dict(self._client_find_one(entity_id))
        except TypeError:
            return None

    def delete_one(self, entity_id: str) -> Optional[str]:
        return self._client_delete_one(entity_id)

    @classmethod
    def format_filters(cls, filters):
        for source, target in cls.filter_renames.items():
            if filters.get(source):
                filters[target] = filters.pop(source)
        filtered = {k: v for k, v in filters.items() if v is not None}
        filters.clear()
        filters.update(filtered)

    @classmethod
    def format_sort(cls, sort):
        if sort:
            for source, target in cls.sort_renames.items():
                if source in sort:
                    sort = sort.replace(source, target)
        return sort


class OrganizationService(BaseService):
    entity_class = Organization
    entity_name = "organization"
    filter_renames = {
        "badge": "badges",
    }
    facets = [
        TermsFacet("producer_type", "producer_type"),
    ]


class DatasetService(BaseService):
    entity_class = Dataset
    entity_name = "dataset"
    filter_renames = {
        "tag": "tags",
        "badge": "badges",
        "topic": "topics",
        "geozone": "geozones",
        "schema_": "schema",
        "organization_badge": "organization_badges",
        "organization": "organization_id_with_name",
    }
    facets = [
        TermsFacet("format_family", "format_family"),
        TermsFacet("access_type", "access_type"),
        TermsFacet("producer_type", "producer_type"),
        TermsFacet("organization_id_with_name", "organization_with_id"),
        TermsFacet("tag", "tags"),
        TermsFacet("license", "license"),
        TermsFacet("format", "format"),
        TermsFacet("schema", "schema"),
        TermsFacet("geozone", "geozones"),
        TermsFacet("granularity", "granularity"),
        TermsFacet("badge", "badges"),
        TermsFacet("topics", "topics"),
        DateRangeFacet("last_update", "last_update"),
    ]

    @classmethod
    def format_filters(cls, filters):
        if filters.get("temporal_coverage"):
            parts = filters.pop("temporal_coverage")
            filters["temporal_coverage_start"] = parts[:10]
            filters["temporal_coverage_end"] = parts[11:]
        super().format_filters(filters)


class ReuseService(BaseService):
    entity_class = Reuse
    entity_name = "reuse"
    filter_renames = {
        "tag": "tags",
        "badge": "badges",
        "organization_badge": "organization_badges",
        "organization": "organization_id_with_name",
    }
    facets = [
        TermsFacet("producer_type", "producer_type"),
        TermsFacet("organization_id_with_name", "organization_with_id"),
        TermsFacet("topic", "topic"),
        TermsFacet("type", "type"),
        TermsFacet("tag", "tags"),
        TermsFacet("badge", "badges"),
        DateRangeFacet("last_update", "last_modified"),
    ]


class DataserviceService(BaseService):
    entity_class = Dataservice
    entity_name = "dataservice"
    filter_renames = {
        "tag": "tags",
        "badge": "badges",
        "topic": "topics",
        "organization": "organization_id_with_name",
    }
    facets = [
        TermsFacet("access_type", "access_type"),
        TermsFacet("producer_type", "producer_type"),
        TermsFacet("organization_id_with_name", "organization_with_id"),
        TermsFacet("tag", "tags"),
        TermsFacet("badge", "badges"),
        DateRangeFacet("last_update", "metadata_modified_at"),
    ]


class TopicService(BaseService):
    entity_class = Topic
    entity_name = "topic"
    facets = [
        TermsFacet("tag", "tags"),
        TermsFacet("organization_id_with_name", "organization_with_id"),
        TermsFacet("producer_type", "producer_type"),
        DateRangeFacet("last_update", "last_modified"),
    ]


class DiscussionService(BaseService):
    entity_class = Discussion
    entity_name = "discussion"
    sort_renames = {
        "created": "created_at",
        "closed": "closed_at",
    }
    facets = [
        TermsFacet("object_type", "subject_class"),
        DateRangeFacet("last_update", "created_at"),
    ]


class PostService(BaseService):
    entity_class = Post
    entity_name = "post"
    facets = [
        DateRangeFacet("last_update", "last_modified"),
    ]
