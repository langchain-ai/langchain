import datetime
import glob
import logging
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, List, Union

if TYPE_CHECKING:
    import vowpal_wabbit_next as vw

logger = logging.getLogger(__name__)


class ModelRepository:
    def __init__(
        self,
        folder: Union[str, os.PathLike],
        with_history: bool = True,
        reset: bool = False,
    ):
        self.folder = Path(folder)
        self.model_path = self.folder / "latest.vw"
        self.with_history = with_history
        if reset and self.has_history():
            logger.warning(
                "There is non empty history which is recommended to be cleaned up"
            )
            if self.model_path.exists():
                os.remove(self.model_path)

        self.folder.mkdir(parents=True, exist_ok=True)

    def get_tag(self) -> str:
        return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    def has_history(self) -> bool:
        return len(glob.glob(str(self.folder / "model-????????-??????.vw"))) > 0

    def save(self, workspace: "vw.Workspace") -> None:
        with open(self.model_path, "wb") as f:
            logger.info(f"storing rl_chain model in: {self.model_path}")
            f.write(workspace.serialize())
        if self.with_history:  # write history
            shutil.copyfile(self.model_path, self.folder / f"model-{self.get_tag()}.vw")

    def load(self, commandline: List[str]) -> "vw.Workspace":
        try:
            import vowpal_wabbit_next as vw
        except ImportError as e:
            raise ImportError(
                "Unable to import vowpal_wabbit_next, please install with "
                "`pip install vowpal_wabbit_next`."
            ) from e

        model_data = None
        if self.model_path.exists():
            with open(self.model_path, "rb") as f:
                model_data = f.read()
        if model_data:
            logger.info(f"rl_chain model is loaded from: {self.model_path}")
            return vw.Workspace(commandline, model_data=model_data)
        return vw.Workspace(commandline)
