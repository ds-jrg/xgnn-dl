import copy
class ClassExpression():
    """
    This class is the base-class for all class expressions. 
    It does not do anything.
    """
    def __init__(self) -> None:
        self.number_classes = 0
        self.number_intersections = 0
        self.number_unions = 0
        pass
    
    def list_init(self, classes):
        content = []
        if isinstance(classes, list):
            if len(classes) > 0:
                for class_ in classes:
                    if isinstance(class_, ClassExpression):
                        content.append(class_)
                        if isinstance(class_, ClassClassExpression):
                            self.number_classes += 1
                        elif isinstance(class_, ClassIntersection):
                            self.number_intersections += 1
                        elif isinstance(class_, ClassUnion):
                            self.number_unions += 1
                    else:
                        raise ValueError(f"{class_} is not a class expression")
        if isinstance(classes, ClassClassExpression):
            content.append(classes)
            self.number_classes += 1
        sorted_content = sorted(content, key=lambda x: 1 if isinstance(x, ClassClassExpression) else 0)
        content = sorted_content
        return content
    
class ClassClassExpression(ClassExpression):
    """
    This saves a class of a class expression
    """
    def __init__(self, name) -> None:
        super().__init__()
        self.class_ = name
        self.number_classes += 1
        pass
    
class ClassIntersection(ClassExpression):
    """
    This class saves an intersection of classes
    Additional Functions:
    - intersect : intersects the class expression with another class expression
    - add to intersection
    - remove from intersection
    - get / set: content
    """
    def __init__(self, classes: list) -> None:
        super().__init__()
        self._content = self.list_init(classes)
        
    @property
    def content(self):
        """Getter for content"""
        return self._content

    @content.setter
    def content(self, value):
        """Setter for content"""
        if isinstance(value, list):
            for class_ in value:
                if not isinstance(class_, ClassExpression):
                    raise ValueError(f"{class_} is not a class expression")
            self._content = self.list_init(value)
        else:
            if not isinstance(value, ClassExpression):
                raise ValueError("Content must be a list or a class expression")
            else:
                self._content = self.list_init([value])
            raise ValueError("Content must be a list")  
        
        
class ClassUnion(ClassExpression):
    """ 
    This class saves a union of classes. It has one list of classes in this union
    Additional functions:
    - add classes
    - remove classes
    """
    def __init__(self, classes: list) -> None:
        super().__init__()
        self._content = self.list_init(classes)
     
    
    @property
    def content(self):
        """Getter for content"""
        return self._content
    @content.setter
    def content(self, value):
        """Setter for content"""
        if isinstance(value, list):
            for class_ in value:
                if not isinstance(class_, ClassExpression):
                    raise ValueError(f"{class_} is not a class expression")
            self._content = self.list_init(value)
        else:
            if not isinstance(value, ClassExpression):
                raise ValueError("Content must be a list or a class expression")
            else:
                self._content = self.list_init([value])
            raise ValueError("Content must be a list")
    