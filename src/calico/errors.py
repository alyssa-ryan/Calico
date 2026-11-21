"""Exceptions the user can act on.

Every error that reaches the CLI or the web UI is a ``CalicoError`` whose message
says what went wrong *and what to do about it*. Anything else is a bug.
"""


class CalicoError(Exception):
    """Base class for expected, user-actionable failures."""


class ConfigError(CalicoError):
    """The settings file or a collection definition is invalid."""


class HomeInsideArchiveError(ConfigError):
    """The calico home is inside a collection root, or holds one."""


class NotInCollectionError(CalicoError):
    """A folder or file is outside every configured collection."""


class UnknownPluginError(CalicoError):
    """A task, model or extractor name is not registered."""


class InvalidRequestError(CalicoError):
    """A selection request is malformed for the task it names."""


class ExtractorError(CalicoError):
    """A feature extractor failed on every photo it was given, or returned the wrong shape:
    the model, not the photos, is at fault."""


class ImageDecodeError(CalicoError):
    """An image file could not be read or decoded."""


class CapabilityError(CalicoError):
    """The configured embedder cannot do what was asked (e.g. encode text)."""


class FolderNotFoundError(CalicoError):
    """A requested folder does not exist (or its drive is not mounted)."""


class RunNotFoundError(CalicoError):
    """No saved run has the requested id."""


class RunUnreadableError(CalicoError):
    """A saved run exists but this version of Calico cannot read it."""


class CatalogTooNewError(CalicoError):
    """The catalog was written by a newer Calico, whose schema this one does not know."""


class CatalogBusyError(CalicoError):
    """Another Calico process held the catalog's write lock for too long."""

    def __init__(self) -> None:
        super().__init__(
            "Another Calico process is writing to the catalog (an index, perhaps, or a "
            "second calico serve). Try again in a moment, or wait until it has finished."
        )


class LabelSetNotFoundError(CalicoError):
    """No saved labelling set has the requested id."""


class LabelSetUnreadableError(CalicoError):
    """A saved labelling set exists but this version of Calico cannot read it."""


class LegacyRatingsError(CalicoError):
    """A ratings file from the previous system cannot be read, or holds no ratings."""


class MigrationError(CalicoError):
    """The old ratings could not all be written to the ratings journal."""


class HoldoutError(CalicoError):
    """The hold-out cannot be read, or would be drawn a second time."""


class BackupError(CalicoError):
    """A backup cannot be made where asked, or a backup file cannot be read."""


class DiskSpaceError(CalicoError):
    """The disk holding the calico home has too little free space for a run's scratch files."""


class TasteUnreadableError(CalicoError):
    """The saved taste exists but cannot be read (a damaged or half-written file)."""


class ControlsError(CalicoError):
    """The pool of control photos cannot be read, or would be drawn a second time."""
