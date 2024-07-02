class Memory:
    """
    """
    def __init__(self):
        self.short_term = []
        self.long_term = []

    @property
    def empty(self):
        return len(self.short_term)==0 and len(self.long_term)==0

    def add(self, txt, type_='short'):
        if type_=='short':
            self.short_term.append(txt.strip())
        elif type_=='long':
            self.long_term.append(txt.strip())
        else:
            raise NotImplementedError(type_)

    def to_text(self):
        txt = ''
        if len(self.short_term)>0:
            txt_ = '\n\n'.join([f'{i+1}: {x}' for i,x in enumerate(self.short_term)])
            txt += f'\n\nHere is what you know specifically, as a numbered list:\n\n{txt_}'
        if len(self.long_term)>0:
            txt_ = '\n\n'.join([f'{i+1}: {x}' for i,x in enumerate(self.long_term)])
            txt += f'\n\nHere is what you know in general, as a numbered list:\n\n{txt_}'
        return txt.strip()

    def consolidate(self):
        #TODO convert short-term memory into long-term memory
        #llm.ask('Summarize task-independent knowledge...')
        #llm.ask('What can I improve...')

        # clear short-term memory
        self.short_term.clear()

