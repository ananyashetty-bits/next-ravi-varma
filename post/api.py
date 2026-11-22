"""The real post module for the app: `APP_POST_MODULE=post.api` (default stays post.stub until the S3 rule is applied)."""
from post.hires import hires            # noqa: F401
from post.restore import restore_faces  # noqa: F401
