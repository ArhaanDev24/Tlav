import os
import tempfile

os.environ.setdefault("TLAV_HOME", tempfile.mkdtemp(prefix="tlav-test-"))
