from udata.core.organization import constants as org_constants
from udata.core.organization.factories import OrganizationFactory
from udata.core.spatial.factories import SpatialCoverageFactory
from udata.core.topic.factories import TopicFactory
from udata.core.topic.search import TopicSearch
from udata.tests.api import APITestCase
from udata.utils import to_iso_datetime


class TestTopicSearch(APITestCase):
    def test_adapter_serialize(self):
        org = OrganizationFactory(name="orga")
        org.add_badge(org_constants.CERTIFIED)
        org.add_badge(org_constants.PUBLIC_SERVICE)
        assert org.public_service is True
        spatial = SpatialCoverageFactory()
        topic = TopicFactory(private=False, organization=org, spatial=spatial)

        assert TopicSearch.is_indexable(topic)
        serialized = TopicSearch.serialize(topic)
        assert serialized == {
            "id": str(topic.id),
            "name": topic.name,
            "description": topic.description,
            "created_at": to_iso_datetime(topic.created_at),
            "last_modified": to_iso_datetime(topic.last_modified),
            "featured": topic.featured,
            "organization": {
                "id": str(topic.organization.id),
                "name": "orga",
                "public_service": 1,
                "followers": 0,
            },
            "owner": None,
            "tags": topic.tags,
            "extras": topic.extras,
            "elements_titles": " ".join([element.title for element in topic.elements]),
            "geozones": [
                {
                    "id": spatial.zones[0].id,
                    "name": spatial.zones[0].name,
                }
            ],
            "granularity": spatial.granularity,
        }
