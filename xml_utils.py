# =========================================================
# @purpose: some utils about xml
# @date：   2019/12
# @version: v1.0
# @author： Xu Huasheng
# @github： https://github.com/xuhuasheng/utils_of_reading_writing_XML
# =========================================================

import xml.etree.ElementTree as ET


def get_elementTree(xml_filePath):
    """
    purpose: get the element tree of xml file
    parameter: xml_filePath, the path to xml file
    return: the element tree of xml file
    """
    tree = ET.parse(xml_filePath) 
    return tree


def get_elements(root, childElementName):
    """
    parameters：
        root: 根节点  
        childElementName: 字节点tag名称
    return:
        elements:根节点下所有符合的子元素对象    
    """
    elements = root.findall(childElementName)
    return elements


def get_element(root, childElementName):
    """
    parameters：
        root: 根节点  
        childElementName: 字节点tag名称
    return:
        elements:根节点下第一个符合的子元素对象    
    """
    element = root.find(childElementName)
    return element


def create_elementTree(root):
    """
    purpose: 以根元素创建elementtree对象
    """
    tree = ET.ElementTree(root)  
    return tree


def create_element(tag, property_map, content):
    """
    parameters：
        tag: 元素tag名称  
        property_map: 元素属性的键值对
        content:元素文本内容
    return:
        @element:新建的元素对象   
    """
    element = ET.Element(tag, property_map)
    element.text = content
    return element

 
def add_childElement(parentElement, childElement):
    """
    purpose: 添加子节点
    parameters：
        parentElement: 父节点  
        childElement: 子节点
    """     
    parentElement.append(childElement)


def formatXml(element, indent, newline, level = 0): 
    """
    purpose: prettify the format of xml for the convenience of reading
    parameters: 
        elemnt: 为传进来的Elment类, 一般为xml根元素
        indent: 用于缩进, 一般为 \t
        newline: 用于换行, linux下为\n, windows下为\r\n
    example:
        formatXml(root, '\t', '\n')  
        import xml.etree.ElementTree as ET
        tree = ET.ElementTree(root)  
        tree.write('*.xml')            
    """
    if element:  # 判断element是否有子元素  
        if element.text == None or element.text.isspace(): # 如果element的text没有内容  
            element.text = newline + indent * (level + 1)    
        else:  
            element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * (level + 1)  
    #else:  # 此处两行如果把注释去掉，Element的text也会另起一行  
        #element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * level  
    temp = list(element) # 将elemnt转成list  
    for subelement in temp:  
        if temp.index(subelement) < (len(temp) - 1): # 如果不是list的最后一个元素，说明下一个行是同级别元素的起始，缩进应一致  
            subelement.tail = newline + indent * (level + 1)  
        else:  # 如果是list的最后一个元素， 说明下一行是母元素的结束，缩进应该少一个  
            subelement.tail = newline + indent * level  
        formatXml(subelement, indent, newline, level = level + 1) # 对子元素进行递归操作