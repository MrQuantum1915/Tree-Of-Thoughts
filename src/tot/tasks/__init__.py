def get_task(name):
    if name == "text":
        from tot.tasks.text import TextTask
        return TextTask()
    if name == "game24":
        from tot.tasks.game24 import Game24Task
        return Game24Task()
    if name == "crosswords":
        from tot.tasks.crosswords import MiniCrosswordsTask
        return MiniCrosswordsTask()
    raise NotImplementedError(f"Unknown task: {name}")
