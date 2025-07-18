from flask import current_app, request
from flask_login import current_user
from werkzeug.exceptions import BadRequest

from udata.api import API, api, fields
from udata.auth import admin_permission
from udata.core.dataservices.models import Dataservice
from udata.core.dataset.api_fields import dataset_fields, dataset_ref_fields
from udata.core.dataset.permissions import OwnablePermission
from udata.core.organization.api_fields import org_ref_fields
from udata.core.organization.permissions import EditOrganizationPermission
from udata.core.user.api_fields import user_ref_fields

from . import actions
from .forms import HarvestSourceForm, HarvestSourceValidationForm
from .models import (
    HARVEST_ITEM_STATUS,
    HARVEST_JOB_STATUS,
    VALIDATION_ACCEPTED,
    VALIDATION_STATES,
    HarvestJob,
    HarvestSource,
)

ns = api.namespace("harvest", "Harvest related operations")


def backends_ids():
    return [b.name for b in actions.list_backends()]


error_fields = api.model(
    "HarvestError",
    {
        "created_at": fields.ISODateTime(
            description="The error creation date", required=True, readonly=True
        ),
        "message": fields.String(description="The error short message", required=True),
        "details": fields.Raw(
            attribute=lambda o: o.details if admin_permission else None,
            description="Optional details (only for super-admins)",
            readonly=True,
        ),
    },
)


log_fields = api.model(
    "HarvestError",
    {
        "level": fields.String(required=True),
        "message": fields.String(required=True),
    },
)


item_fields = api.model(
    "HarvestItem",
    {
        "remote_id": fields.String(description="The item remote ID to process", required=True),
        "dataset": fields.Nested(
            dataset_ref_fields, description="The processed dataset", allow_null=True
        ),
        "dataservice": fields.Nested(
            Dataservice.__ref_fields__, description="The processed dataservice", allow_null=True
        ),
        "status": fields.String(
            description="The item status", required=True, enum=list(HARVEST_ITEM_STATUS)
        ),
        "created": fields.ISODateTime(description="The item creation date", required=True),
        "started": fields.ISODateTime(description="The item start date"),
        "ended": fields.ISODateTime(description="The item end date"),
        "errors": fields.List(fields.Nested(error_fields), description="The item errors"),
        "logs": fields.List(fields.Nested(log_fields), description="The item logs"),
        "args": fields.List(fields.String, description="The item positional arguments", default=[]),
        "kwargs": fields.Raw(description="The item keyword arguments", default={}),
    },
)

job_fields = api.model(
    "HarvestJob",
    {
        "id": fields.String(description="The job execution ID", required=True),
        "created": fields.ISODateTime(description="The job creation date", required=True),
        "started": fields.ISODateTime(description="The job start date"),
        "ended": fields.ISODateTime(description="The job end date"),
        "status": fields.String(
            description="The job status", required=True, enum=list(HARVEST_JOB_STATUS)
        ),
        "errors": fields.List(
            fields.Nested(error_fields), description="The job initialization errors"
        ),
        "items": fields.List(fields.Nested(item_fields), description="The job collected items"),
        "source": fields.String(description="The source owning the job", required=True),
    },
)

job_page_fields = api.model("HarvestJobPage", fields.pager(job_fields))

validation_fields = api.model(
    "HarvestSourceValidation",
    {
        "state": fields.String(
            description="Is it validated or not", enum=list(VALIDATION_STATES), required=True
        ),
        "by": fields.Nested(
            user_ref_fields,
            allow_null=True,
            readonly=True,
            description="Who performed the validation",
        ),
        "on": fields.ISODateTime(
            readonly=True, description="Date date on which validation was performed"
        ),
        "comment": fields.String(
            description="A comment about the validation. Required on rejection"
        ),
    },
)

source_fields = api.model(
    "HarvestSource",
    {
        "id": fields.String(description="The source unique identifier", readonly=True),
        "name": fields.String(description="The source display name", required=True),
        "description": fields.Markdown(description="The source description"),
        "url": fields.String(description="The source base URL", required=True),
        "backend": fields.String(
            description="The source backend", enum=backends_ids, required=True
        ),
        "config": fields.Raw(description="The configuration as key-value pairs"),
        "created_at": fields.ISODateTime(
            description="The source creation date", required=True, readonly=True
        ),
        "active": fields.Boolean(description="Is this source active", required=True, default=False),
        "autoarchive": fields.Boolean(
            description="If enabled, datasets not present on the remote source will be automatically archived",  # noqa
            required=True,
            default=True,
        ),
        "validation": fields.Nested(
            validation_fields, readonly=True, description="Has the source been validated"
        ),
        "last_job": fields.Nested(
            job_fields, description="The last job for this source", allow_null=True, readonly=True
        ),
        "owner": fields.Nested(
            user_ref_fields, allow_null=True, description="The owner information"
        ),
        "organization": fields.Nested(
            org_ref_fields, allow_null=True, description="The producer organization"
        ),
        "deleted": fields.ISODateTime(description="The source deletion date", readonly=True),
        "schedule": fields.String(
            description="The source schedule (interval or cron expression)", readonly=True
        ),
    },
)

source_page_fields = api.model("HarvestSourcePage", fields.pager(source_fields))


filter_fields = api.model(
    "HarvestFilter",
    {
        "label": fields.String(description="A localized human-readable label"),
        "key": fields.String(description="The filter key"),
        "type": fields.String(description="The filter expected type"),
        "description": fields.String(description="The filter details"),
    },
)

feature_fields = api.model(
    "HarvestFeature",
    {
        "label": fields.String(description="A localized human-readable and descriptive label"),
        "key": fields.String(description="The feature key"),
        "description": fields.String(description="Some details about the behavior"),
        "default": fields.String(description="The feature default state (true is enabled)"),
    },
)

harvest_extra_fields = api.model(
    "HarvestExtraConfig",
    {
        "label": fields.String(description="A localized human-readable and descriptive label"),
        "key": fields.String(description="The config key"),
        "description": fields.String(description="Some details about the behavior"),
        "default": fields.String(description="The config default value"),
    },
)

backend_fields = api.model(
    "HarvestBackend",
    {
        "id": fields.String(description="The backend identifier"),
        "label": fields.String(description="The backend display name"),
        "filters": fields.List(
            fields.Nested(filter_fields), description="The backend supported filters"
        ),
        "features": fields.List(
            fields.Nested(feature_fields), description="The backend optional features"
        ),
        "extra_configs": fields.List(
            fields.Nested(harvest_extra_fields),
            description="The backend extra configuration variables",
        ),
    },
)

preview_dataservice_fields = api.clone(
    "DataservicePreview",
    Dataservice.__ref_fields__,
    {
        "self_web_url": fields.Raw(
            attribute=lambda _d: None, description="The dataservice webpage URL (fake)"
        ),
        "self_api_url": fields.Raw(
            attribute=lambda _d: None, description="The dataservice API URL (fake)"
        ),
    },
)


preview_dataset_fields = api.clone(
    "DatasetPreview",
    dataset_fields,
    {
        "uri": fields.Raw(attribute=lambda _d: None, description="The dataset API URL (fake)"),
        "page": fields.Raw(attribute=lambda _d: None, description="The dataset page URL (fake)"),
    },
)

preview_item_fields = api.clone(
    "HarvestItemPreview",
    item_fields,
    {
        "dataset": fields.Nested(
            preview_dataset_fields, description="The processed dataset", allow_null=True
        ),
        "dataservice": fields.Nested(
            preview_dataservice_fields, description="The processed dataset", allow_null=True
        ),
    },
)

preview_job_fields = api.clone(
    "HarvestJobPreview",
    job_fields,
    {
        "items": fields.List(
            fields.Nested(preview_item_fields), description="The job collected items"
        ),
    },
)

source_parser = api.page_parser()
source_parser.add_argument(
    "owner", type=str, location="args", help="The organization or user ID to filter on"
)
source_parser.add_argument(
    "deleted", type=bool, location="args", default=False, help="Include sources flaggued as deleted"
)
source_parser.add_argument("q", type=str, location="args", help="The search query")


@ns.route("/sources/", endpoint="harvest_sources")
class SourcesAPI(API):
    @api.doc("list_harvest_sources")
    @api.expect(source_parser)
    @api.marshal_list_with(source_page_fields)
    def get(self):
        """List all harvest sources"""
        args = source_parser.parse_args()

        sources = HarvestSource.objects()

        if not args["deleted"]:
            sources = sources.visible()

        if args["owner"]:
            sources = sources.owned_by(args["owner"])

        if args["q"]:
            phrase_query = " ".join([f'"{elem}"' for elem in args["q"].split(" ")])
            sources = sources.search_text(phrase_query)

        return sources.paginate(args["page"], args["page_size"])

    @api.secure
    @api.doc("create_harvest_source")
    @api.expect(source_fields)
    @api.marshal_with(source_fields)
    def post(self):
        """Create a new harvest source"""
        form = api.validate(HarvestSourceForm)
        if form.organization.data:
            EditOrganizationPermission(form.organization.data).test()
        source = actions.create_source(**form.data)
        return source, 201


@ns.route("/source/<string:ident>", endpoint="harvest_source")
@api.param("ident", "A source ID or slug")
class SourceAPI(API):
    @api.doc("get_harvest_source")
    @api.marshal_with(source_fields)
    def get(self, ident):
        """Get a single source given an ID or a slug"""
        return actions.get_source(ident)

    @api.secure
    @api.doc("update_harvest_source")
    @api.expect(source_fields)
    @api.marshal_with(source_fields)
    def put(self, ident):
        """Update a harvest source"""
        source = actions.get_source(ident)
        OwnablePermission(source).test()
        form = api.validate(HarvestSourceForm, source)
        source = actions.update_source(ident, form.data)
        return source

    @api.secure
    @api.doc("delete_harvest_source")
    @api.marshal_with(source_fields)
    def delete(self, ident):
        source: HarvestSource = actions.get_source(ident)
        OwnablePermission(source).test()
        return actions.delete_source(ident), 204


@ns.route("/source/<string:ident>/validate", endpoint="validate_harvest_source")
@api.param("ident", "A source ID or slug")
class ValidateSourceAPI(API):
    @api.doc("validate_harvest_source")
    @api.secure(admin_permission)
    @api.expect(validation_fields)
    @api.marshal_with(source_fields)
    def post(self, ident):
        """Validate or reject an harvest source"""
        form = api.validate(HarvestSourceValidationForm)
        if form.state.data == VALIDATION_ACCEPTED:
            return actions.validate_source(ident, form.comment.data)
        else:
            return actions.reject_source(ident, form.comment.data)


@ns.route("/source/<string:ident>/run", endpoint="run_harvest_source")
@api.param("ident", "A source ID or slug")
class RunSourceAPI(API):
    @api.doc("run_harvest_source")
    @api.secure
    @api.marshal_with(source_fields)
    def post(self, ident):
        enabled = current_app.config.get("HARVEST_ENABLE_MANUAL_RUN")
        if not enabled and not current_user.sysadmin:
            api.abort(
                400,
                "Cannot run source manually. Please contact the platform if you need to reschedule the harvester.",
            )

        source: HarvestSource = actions.get_source(ident)
        OwnablePermission(source).test()

        if source.validation.state != VALIDATION_ACCEPTED:
            api.abort(400, "Source is not validated. Please validate the source before running.")

        actions.launch(ident)

        return source


@ns.route("/source/<string:ident>/schedule", endpoint="schedule_harvest_source")
@api.param("ident", "A source ID or slug")
class ScheduleSourceAPI(API):
    @api.doc("schedule_harvest_source")
    @api.secure(admin_permission)
    @api.expect((str, "A cron expression"))
    @api.marshal_with(source_fields)
    def post(self, ident):
        """Schedule an harvest source"""
        # Handle both syntax: quoted and unquoted
        try:
            data = request.json
        except BadRequest:
            data = request.data.decode("utf-8")
        return actions.schedule(ident, data)

    @api.doc("unschedule_harvest_source")
    @api.secure(admin_permission)
    @api.marshal_with(source_fields)
    def delete(self, ident):
        """Unschedule an harvest source"""
        return actions.unschedule(ident), 204


@ns.route("/source/preview", endpoint="preview_harvest_source_config")
class PreviewSourceConfigAPI(API):
    @api.secure
    @api.expect(source_fields)
    @api.doc("preview_harvest_source_config")
    @api.marshal_with(preview_job_fields)
    def post(self):
        """Preview an harvesting from a source created with the given payload"""
        form = api.validate(HarvestSourceForm)
        if form.organization.data:
            EditOrganizationPermission(form.organization.data).test()
        return actions.preview_from_config(**form.data)


@ns.route("/source/<string:ident>/preview", endpoint="preview_harvest_source")
@api.param("ident", "A source ID or slug")
class PreviewSourceAPI(API):
    @api.secure
    @api.doc("preview_harvest_source")
    @api.marshal_with(preview_job_fields)
    def get(self, ident):
        """Preview a single harvest source given an ID or a slug"""
        return actions.preview(ident)


parser = api.parser()
parser.add_argument("page", type=int, default=1, location="args", help="The page to fetch")
parser.add_argument(
    "page_size", type=int, default=20, location="args", help="The page size to fetch"
)


@ns.route("/source/<string:ident>/jobs/", endpoint="harvest_jobs")
class JobsAPI(API):
    @api.doc("list_harvest_jobs")
    @api.expect(parser)
    @api.marshal_with(job_page_fields)
    def get(self, ident):
        """List all jobs for a given source"""
        args = parser.parse_args()
        qs = HarvestJob.objects(source=ident)
        qs = qs.order_by("-created")
        return qs.paginate(args["page"], args["page_size"])


@ns.route("/job/<string:ident>/", endpoint="harvest_job")
class JobAPI(API):
    @api.doc("get_harvest_job")
    @api.expect(parser)
    @api.marshal_with(job_fields)
    def get(self, ident):
        """List all jobs for a given source"""
        return actions.get_job(ident)


@ns.route("/backends", endpoint="harvest_backends")
class ListBackendsAPI(API):
    @api.doc("harvest_backends")
    @api.marshal_with(backend_fields)
    def get(self):
        """List all available harvest backends"""
        return sorted(
            [
                {
                    "id": b.name,
                    "label": b.display_name,
                    "filters": [f.as_dict() for f in b.filters],
                    "features": [f.as_dict() for f in b.features],
                    "extra_configs": [f.as_dict() for f in b.extra_configs],
                }
                for b in actions.list_backends()
            ],
            key=lambda b: b["label"],
        )


@ns.route("/job_status", endpoint="havest_job_status")
class ListHarvesterAPI(API):
    @api.doc(model=[str])
    def get(self):
        """List all available harvesters"""
        return actions.list_backends()
