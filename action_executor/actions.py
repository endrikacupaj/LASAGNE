class ActionOperator:
    def __init__(self, kg):
        self.kg = kg

    def find(self, e, p):
        if isinstance(e, list):
            return self.find_set(e, p)

        if e is None or p is None:
            return None

        if e not in self.kg.triples['subject'] or p not in self.kg.triples['subject'][e]:
            return set()

        return set(self.kg.triples['subject'][e][p])

    def find_reverse(self, e, p):
        if isinstance(e, list):
            return self.find_reverse_set(e, p)

        if e is None or p is None:
            return None

        if e not in self.kg.triples['object'] or p not in self.kg.triples['object'][e]:
            return set()

        return set(self.kg.triples['object'][e][p])

    def find_set(self, e_set, p):
        result_set = set()
        for e in e_set:
            result_set.update(self.find(e, p))

        return result_set

    def find_reverse_set(self, e_set, p):
        result_set = set()
        for e in e_set:
            result_set.update(self.find_reverse(e, p))

        return result_set

    def filter_type(self, ent_set, typ):
        if type(ent_set) is not set or typ is None:
            return None

        result = set()

        for o in ent_set:
            if (o in self.kg.entity_type and typ in self.kg.entity_type[o]):
                result.add(o)

        return result

    def filter_multi_types(self, ent_set, t1, t2):
        typ_set = {t1, t2}
        if type(ent_set) is not set or type(typ_set) is not set:
            return None

        result = set()

        for o in ent_set:
            if (o in self.kg.entity_type and len(typ_set.intersection(set(self.kg.entity_type[o]))) > 0):
                result.add(o)

        return result

    def find_tuple_counts(self, r, t1, t2):
        if r is None or t1 is None or t2 is None:
            return None

        tuple_count = dict()

        for s in self.kg.triples['relation']['subject'][r]:
            if (s in self.kg.entity_type and t1 in self.kg.entity_type[s]):
                count = 0
                for o in self.kg.triples['relation']['subject'][r][s]:
                    if (o in self.kg.entity_type and t2 in self.kg.entity_type[o]):
                        count += 1

                tuple_count[s] = count

        return tuple_count

    def find_reverse_tuple_counts(self, r, t1, t2):
        if r is None or t1 is None or t2 is None:
            return None

        tuple_count = dict()

        for o in self.kg.triples['relation']['object'][r]:
            if (o in self.kg.entity_type and t1 in self.kg.entity_type[o]):
                count = 0
                for s in self.kg.triples['relation']['object'][r][o]:
                    if (s in self.kg.entity_type and t2 in self.kg.entity_type[s]):
                        count += 1

                tuple_count[o] = count

        return tuple_count

    def greater(self, type_dict, value):
        return set([k for k, v in type_dict.items() if v > value and v >= 0])

    def less(self, type_dict, value):
        return set([k for k, v in type_dict.items() if v < value and v >= 0])

    def equal(self, type_dict, value):
        return set([k for k, v in type_dict.items() if v == value and v >= 0])

    def approx(self, type_dict, value, interval=15):
        # ambiguous action
        # simply check for more than 0
        return set([k for k, v in type_dict.items() if v > 0])

    def atmost(self, type_dict, max_value):
        return set([k for k, v in type_dict.items() if v <= max_value and v >= 0])

    def atleast(self, type_dict, min_value):
        return set([k for k, v in type_dict.items() if v >= min_value and v >= 0])

    def argmin(self, type_dict, value=0):
        min_value = min(type_dict.values())
        return set([k for k, v in type_dict.items() if v == min_value])

    def argmax(self, type_dict, value=0):
        max_value = max(type_dict.values())
        return set([k for k, v in type_dict.items() if v == max_value])

    def is_in(self, ent, set_ent):
        return set(ent).issubset(set_ent)

    def count(self, in_set):
        return len(in_set)

    def union(self, *args):
        if all(isinstance(x, set) for x in args):
            return args[0].union(*args)
        else:
            return {k: args[0].get(k, 0) + args[1].get(k, 0) for k in set(args[0]) | set(args[1])}

    def intersection(self, s1, s2):
        return s1.intersection(s2)

    def difference(self, s1, s2):
        return s1.difference(s2)
