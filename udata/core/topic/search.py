import datetime

from udata.core.spatial.constants import ADMIN_LEVEL_MAX
from udata.core.spatial.models import GeoZone, admin_levels
from udata.core.topic.models import Topic
from udata.core.topic.parsers import TopicApiParser
from udata.models import Organization, User
from udata.search import (
    BoolFilter,
    Filter,
    ListFilter,
    ModelSearchAdapter,
    ModelTermsFilter,
    register,
)
from udata.utils import to_iso_datetime

__all__ = ("TopicSearch",)

DEFAULT_SORTING = "-created_at"


@register
class TopicSearch(ModelSearchAdapter):
    model = Topic
    search_url = "topics/"

    sorts = {"created": "created_at", "last_modified": "last_modified"}

    filters = {
        "tag": ListFilter(),
        "organization": ModelTermsFilter(model=Organization),
        "owner": ModelTermsFilter(model=User),
        "geozone": ModelTermsFilter(model=GeoZone),
        "granularity": Filter(),
        "featured": BoolFilter(),
    }

    @classmethod
    def is_indexable(cls, topic: Topic) -> bool:
        return not topic.private

    @classmethod
    def mongo_search(cls, args):
        topics = Topic.objects.visible()
        topics = TopicApiParser.parse_filters(topics, args)

        sort = (
            cls.parse_sort(args["sort"])
            or ("$text_score" if args["q"] else None)
            or DEFAULT_SORTING
        )
        return topics.order_by(sort).paginate(args["page"], args["page_size"])

    @classmethod
    def serialize(cls, topic: Topic) -> dict:
        organization = None
        owner = None
        if topic.organization:
            org = Organization.objects(id=topic.organization.id).first()
            organization = {
                "id": str(org.id),
                "name": org.name,
                "public_service": 1 if org.public_service else 0,
                "followers": org.metrics.get("followers", 0),
            }
        elif topic.owner:
            owner = User.objects(id=topic.owner.id).first()
        extras = {}
        for key, value in topic.extras.items():
            extras[key] = to_iso_datetime(value) if isinstance(value, datetime.datetime) else value

        def get_elements_titles(elements: list, max_elements: int = 1000):
            titles = []
            for element in elements[:max_elements]:
                if element.title:
                    titles.append(element.title)
                elif element.element and getattr(element.element, "title", None):
                    titles.append(element.element.title)
            return " ".join(titles)

        document = {
            "id": str(topic.id),
            "name": topic.name,
            "description": topic.description,
            "created_at": to_iso_datetime(topic.created_at),
            "last_modified": to_iso_datetime(topic.last_modified),
            "featured": topic.featured,
            "organization": organization,
            "owner": str(owner.id) if owner else None,
            "tags": topic.tags,
            "extras": extras,
            "elements_titles": get_elements_titles(topic.elements),
        }

        if topic.spatial is not None:
            zone_ids = [z.id for z in topic.spatial.zones]
            zones = GeoZone.objects(id__in=zone_ids)
            geozones = []
            coverage_level = ADMIN_LEVEL_MAX
            for zone in zones:
                geozones.append(
                    {
                        "id": zone.id,
                        "name": zone.name,
                    }
                )
                coverage_level = min(coverage_level, admin_levels[zone.level])
            document.update(
                {
                    "geozones": geozones,
                    "granularity": topic.spatial.granularity,
                }
            )

        return document
