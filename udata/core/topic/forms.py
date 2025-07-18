from udata.core.spatial.forms import SpatialCoverageField
from udata.forms import ModelForm, fields, validators
from udata.i18n import lazy_gettext as _

from .models import Topic, TopicElement

__all__ = ("TopicForm", "TopicElementForm")


class TopicElementForm(ModelForm):
    model_class = TopicElement

    title = fields.StringField(_("Title"))
    description = fields.StringField(_("Description"))
    tags = fields.TagField(_("Tags"))
    extras = fields.ExtrasField()
    element = fields.ModelField(_("Element"))

    def validate(self, extra_validators=None):
        """
        Make sure that either title or element is set.
        (Empty nested element is a valid use case for "placeholder" elements)
        """
        validation = super().validate(extra_validators)
        if not self.element.data and not self.title.data:
            self.element.errors.append(_("A topic element must have a title or an element."))
            return False
        return validation


class TopicForm(ModelForm):
    model_class = Topic

    owner = fields.CurrentUserField()
    organization = fields.PublishAsField(_("Publish as"))

    name = fields.StringField(_("Name"), [validators.DataRequired()])
    description = fields.MarkdownField(_("Description"), [validators.DataRequired()])

    elements = fields.NestedModelList(TopicElementForm)

    def save(self, commit=True, **kwargs):
        """Custom save to handle TopicElement creation properly"""
        # Store elements data
        elements_data = self.elements.data

        # Create topic data without elements field (not on model)
        topic_data = {k: v for k, v in self.data.items() if k != "elements"}

        # Get or create topic instance
        if hasattr(self, "instance") and self.instance:
            # Update existing topic
            topic = self.instance
            for key, value in topic_data.items():
                setattr(topic, key, value)
        else:
            # Create new topic
            topic = self.model_class(**topic_data)

        # Save topic first so it can be referenced
        if commit:
            topic.save()

        # Create elements and associate them with the topic
        if elements_data:
            for element_data in elements_data:
                element_form = TopicElementForm(data=element_data)
                if element_form.validate():
                    element = element_form.save(commit=False)
                    element.topic = topic
                    if commit:
                        element.save()

        return topic

    spatial = SpatialCoverageField(
        _("Spatial coverage"), description=_("The geographical area covered by the data.")
    )

    tags = fields.TagField(_("Tags"), [validators.DataRequired()])
    private = fields.BooleanField(_("Private"))
    featured = fields.BooleanField(_("Featured"))
    extras = fields.ExtrasField()
