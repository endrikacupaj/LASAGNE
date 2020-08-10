from unidecode import unidecode
class NERBase:
    def __init__(self, kg, preprocessed_data, tokenizer):
        self.kg = kg
        self.preprocessed_data = preprocessed_data
        self.tokenizer = tokenizer

    def find_entity_in_utterance(self, entity, utterance):
        results = []
        ent_len = len(entity)
        for ind in (i for i, e in enumerate(utterance) if e == entity[0]):
            if utterance[ind:ind+ent_len] == entity:
                results.append((ind, ind+ent_len-1))
        return results

    def new_direct_question(self, user, system, is_verification=False):
        utterance = unidecode(user['utterance'].lower())
        entities = user['entities_in_utterance']

        context = self.tokenizer(utterance)
        ent_data = {}
        for entity in entities:
            ent_label = self.kg.id_entity[entity]
            tok_ent = self.tokenizer(unidecode(ent_label.lower()))
            try:
                ent_in_utter = self.find_entity_in_utterance(tok_ent, context)
            except Exception as ex:
                print(ex)
                self.log_error('Entity not found on utterance', user['utterance'])
                user['is_ner_spurious'] = True
                system['is_ner_spurious'] = True
                return user, system
            if not is_verification and not ent_in_utter:
                self.log_error('Entity not found on utterance', user['utterance'])
                user['is_ner_spurious'] = True
                system['is_ner_spurious'] = True
                return user, system
            ent_data[entity] = {
                'type': self.get_type(entity),
                'ent_label': ent_label,
                'tok_ent': tok_ent,
                'indices': list(range(ent_in_utter[0][0], ent_in_utter[0][1]+1)) if ent_in_utter else []
            }

        utter_context = []
        for i, word in enumerate(context):
            word_context = [i, word, 'NA', 'NA', 'O']
            for ent, data in ent_data.items():
                if i in data['indices']:
                    word_context[-1] = 'B' if data['indices'] and i == data['indices'][0] else 'I'
                    word_context[-2] = data['type']
                    word_context[-3] = ent
            utter_context.append(word_context)

        user['context'] = utter_context
        system['context'] = self.get_system_context(system)
        user['is_ner_spurious'] = False
        system['is_ner_spurious'] = False

        return user, system

    def direct_question(self, user, system):
        # get data
        data = self.extract_data_from_direct(user, system)

        if not data:
            user['is_ner_spurious'] = True
            system['is_ner_spurious'] = True
            return user, system

        utterance = data['utterance']
        context = data['context']
        entities = data['entities']

        # get entity texts from tokenized utterance and create IOB-tags
        ent_tags = []
        seen_ents = set()
        for entity in entities:
            assert entity in context, utterance
            ent_type = self.get_type(entity)
            ent_idx = context.index(entity)
            before_ent_idx = ent_idx-1 # context.index(context[ent_idx-1])
            word_after_ent = context[ent_idx+1]
            new_context = []
            for word_idx, word in enumerate(utterance):
                if word_idx <= before_ent_idx:
                    new_context.append(word)
                    continue
                if word == word_after_ent:
                    new_context.extend(context[ent_idx+1:])
                    context = new_context.copy()
                    break
                if entity not in seen_ents:
                    seen_ents.add(entity)
                    ent_tags.append([word_idx, word, entity, ent_type, 'B'])
                else:
                    ent_tags.append([word_idx, word, entity, ent_type, 'I'])
                new_context.append(word)

        # final ner list
        tag_idx = [tag[0] for tag in ent_tags]
        ner_tags = []
        for i, word_utter in enumerate(utterance):
            if i not in tag_idx:
                ner_tags.append([i, word_utter, 'NA', 'NA', 'O'])
                continue
            for tag in ent_tags:
                if i == tag[0]:
                    ner_tags.append(tag)

        if len(ner_tags) != len(utterance):
            # something went wrong, write example on txt file
            self.log_error('No match', user["utterance"])
            user['is_ner_spurious'] = True
            system['is_ner_spurious'] = True
            return user, system

        user['context'] = ner_tags
        system['context'] = self.get_system_context(system)
        user['is_ner_spurious'] = False
        system['is_ner_spurious'] = False

        return user, system

    def indirect_question(self, user, system):
        user['context'] = [[i, word, 'NA', 'NA', 'O'] for i, word in enumerate(self.tokenizer(user['utterance'].lower()))]
        system['context'] = self.get_system_context(system)
        user['is_ner_spurious'] = False
        system['is_ner_spurious'] = False
        return user, system

    def ellipsis_question(self, user, system, key_word='about'):
        # check entity type
        entity = user['entities_in_utterance'][0]
        ent_type = self.get_type(entity)

        # get about index for identifying the entity
        tok_utterance = self.tokenizer(user['utterance'].lower())
        key_word_idx = tok_utterance.index(key_word)

        # create ner tags
        ner_tags = []
        for i, word in enumerate(tok_utterance):
            if i <= key_word_idx:
                ner_tags.append([i, word, 'NA', 'NA', 'O'])
            elif i == key_word_idx + 1:
                ner_tags.append([i, word, entity, ent_type, 'B'])
            else:
                ner_tags.append([i, word, entity, ent_type, 'I'])

        user['context'] = ner_tags
        system['context'] = self.get_system_context(system)
        user['is_ner_spurious'] = False
        system['is_ner_spurious'] = False

        return user, system

    def clarification_question(self, user, system):
        # get about index for identifying the entity
        tok_utterance = self.tokenizer(user['utterance'].lower())

        # create ner tags
        ner_tags = []
        if user['utterance'] == 'Yes':
            for i, word in enumerate(tok_utterance):
                ner_tags.append([i, word, 'NA', 'NA', 'O'])
        else:
            # check entity type
            entity = user['entities'][0]
            ent_type = self.get_type(entity)
            meant_idx = tok_utterance.index('meant')
            dot_idx = tok_utterance.index('could')
            for i, word in enumerate(tok_utterance):
                if i <= meant_idx or i >= dot_idx:
                    ner_tags.append([i, word, 'NA', 'NA', 'O'])
                elif i == meant_idx + 1:
                    ner_tags.append([i, word, entity, ent_type, 'B'])
                else:
                    ner_tags.append([i, word, entity, ent_type, 'I'])

        user['context'] = ner_tags
        system['context'] = self.get_system_context(system)
        user['is_ner_spurious'] = False
        system['is_ner_spurious'] = False

        return user, system

    def extract_data_from_direct(self, user, system):
        # get index from preprocessed
        index = self.get_index_from_preprocessed(user)
        if not index: # if no index then skip
            return {}

        # get data based on index
        utterance = self.preprocessed_data['utterances'][index]
        context = self.preprocessed_data['contexts'][index]
        entities = self.preprocessed_data['entities'][index]

        if not entities: # if no index then skip
            return {}

        entities = entities.split('|') if entities != '' else []
        tok_utterance = self.tokenizer(utterance)

        # we do not want the tokenizer to split wikidata ids for the context utterance
        tok_context = []
        ordered_entities = []
        for cont in context.split():
            if cont in entities:
                tok_context.append(cont)
                ordered_entities.append(cont)
            else:
                tok_context.extend(self.tokenizer(cont))

        # TODO check for Simple Question|Mult. Entity|Indirect

        return {
            'utterance': tok_utterance,
            'context': tok_context,
            'entities': ordered_entities,
        }

    def get_index_from_preprocessed(self, user):
        # some entities contain unicode characters
        # we identify them by including one by one the characters in the utterance
        if user['utterance'] in self.preprocessed_data['utterances']:
            return self.preprocessed_data['utterances'].index(user['utterance'])
        else:
            index = None
            filtered_utterances = self.preprocessed_data['utterances']
            index_to_remove = []
            user_utterance = ''
            # search forward
            for i in range(len(user['utterance'])):
                user_utterance += user['utterance'][i]
                # user_utterance = user['utterance'][:i]
                # for itr in reversed(index_to_remove):
                #     user_utterance = user_utterance[:itr] + user_utterance[itr+1:]
                results = [utterance for utterance in filtered_utterances if utterance.startswith(user_utterance)]
                if len(results) == 1:
                    index = self.preprocessed_data['utterances'].index(results[0])
                    break
                elif len(results) == 0:
                    user_utterance = user_utterance[:-1]
                    # index_to_remove.append(i)
                else:
                    filtered_utterances = results
            # search backward
            # if not index:
            #     filtered_utterances = self.preprocessed_data['utterances']
            #     index_to_remove = []
            #     for i in range(len(user['utterance'])):
            #         user_utterance = user['utterance'][::-1][:i]
            #         for itr in reversed(index_to_remove):
            #             user_utterance = user_utterance[:itr] + user_utterance[itr+1:]
            #         results = [utterance for utterance in filtered_utterances if utterance[::-1].startswith(user_utterance)]
            #         if len(results) == 1:
            #             index = self.preprocessed_data['utterances'].index(results[0])
            #             break
            #         elif len(results) == 0:
            #             index_to_remove.append(i)
            #         else:
            #             filtered_utterances = results

            if not index:
                self.log_error('No preprocessed', user["utterance"])
                return None

            entities = self.preprocessed_data['entities'][index]
            entities = entities.split('|') if entities != '' else []
            if entities != user['entities_in_utterance']: # if no index then skip
                self.log_error('No entity match', user["utterance"])
                return None

            return index

    def get_system_context(self, system):
        ner_tags = []
        if len(system['entities_in_utterance']) > 0:
            idx_counter = 0
            for j, entity in enumerate(system['entities_in_utterance']):
                ent_type = self.get_type(entity)
                label = self.kg.id_entity[entity]
                for i, word in enumerate(self.tokenizer(label.lower())):
                    if i == 0:
                        ner_tags.append([idx_counter + i, word, entity, ent_type, 'B'])
                    else:
                        ner_tags.append([idx_counter + i, word, entity, ent_type, 'I'])

                if j+1 < len(system['entities_in_utterance']):
                    ner_tags.append([idx_counter + i + 1, ',', 'NA', 'NA', 'O'])
                    idx_counter = idx_counter + i + 2
        elif system['utterance'].isnumeric():
            ner_tags.append([0, 'num', 'NA', 'NA', 'O'])
        elif system['utterance'] == 'YES':
            ner_tags.append([0, 'yes', 'NA', 'NA', 'O'])
        elif system['utterance'] == 'NO':
            ner_tags.append([0, 'no', 'NA', 'NA', 'O'])
        elif system['utterance'] == 'YES and NO respectively':
            ner_tags.append([0, 'no', 'NA', 'NA', 'O'])
        elif system['utterance'] == 'NO and YES respectively':
            ner_tags.append([0, 'no', 'NA', 'NA', 'O'])

        return ner_tags

    def get_type(self, entity):
        if entity not in self.kg.entity_type:
            self.log_error('No type', entity)
            return 'NA'
        else:
            return self.kg.entity_type[entity][0]

    def log_error(self, txt, context):
        unicoce_txt = open('TOFIX.txt', 'a')
        unicoce_txt.write(f'{self.__class__.__name__}\t{txt}\t{context}\n')
        unicoce_txt.close()