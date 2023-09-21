class ClassExpression():
    """
    This class is the base-class for all class expressions. 
    It does not do anything.
    """
    def __init__(self) -> None:
        pass
    
class ClassClassExpression:
    """
    This saves a class of a class expression
    """
    def __init__(self, name) -> None:
        self.class = name
        pass