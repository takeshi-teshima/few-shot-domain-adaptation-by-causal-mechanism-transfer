import re


def tex_escape(text):
    """
    :param text: a plain text message
    :return: the message escaped to appear correctly in LaTeX
    """
    conv = {
        #         '&': r'\\&',
        #         '%': r'\\%',
        #         '$': r'\\$',
        #         '#': r'\\#',
        '_': r'\\_',
        #         '{': r'\\{',
        #         '}': r'\\}',
        #         '~': r'\\textasciitilde{}',
        #         '^': r'\\^{}',
        #         '\\\\': r'\\textbackslash{}',
        #         '<': r'\\textless{}',
        #         '>': r'\\textgreater{}',
    }
    regex = re.compile('|'.join(
        re.escape(str(key))
        for key in sorted(conv.keys(), key=lambda item: -len(item))))
    return regex.sub(lambda match: conv[match.group()], text)
