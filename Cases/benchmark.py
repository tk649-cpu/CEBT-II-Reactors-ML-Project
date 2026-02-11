class Benchmark:
    """Base class for reaction process benchmarks.
    """

    def __init__(
        self,
        structure_params,
        physics_params,
        reaction_params,
        transport_params,
        operation_params,
        operation_name2ind,
        measure_ind2name,
        var2unit
    ):
        self._structure_params = structure_params
        self._physics_params = physics_params
        self._reaction_params = reaction_params
        self._transport_params = transport_params
        self._operation_params = operation_params
        self._operation_name2ind = operation_name2ind
        self._measure_ind2name = measure_ind2name
        self._var2unit = var2unit
    
    def _param_dict2list(self, param_dict):
        param_list = []
        for k, v in param_dict.items():
            _param_dict = {}
            _param_dict["param"] = k[0]
            if k[1] and k[1] in self.solids:
                _param_dict["solid"] = k[1]
            if k[1] and k[1] in self.gases:
                _param_dict["gas"] = k[1]
            if k[2] and k[2] in self.streams:
                _param_dict["stream"] = k[2]
            if k[3] and k[3] in self.reactions:
                _param_dict["reaction"] = k[3]
            if k[4] and k[4] in self.species:
                _param_dict["species"] = k[4]
            param_list.append((_param_dict, v))
        return param_list

    def _param_list2dict(self, param_list):
        param_dict = {}
        for k, v in param_list:
            _param_tuple = []
            _param_tuple.append(k["param"])
            if "solid" in k:
                _param_tuple.append(k["solid"])
            elif "gas" in k:
                _param_tuple.append(k["gas"])
            else:
                _param_tuple.append(None)
            _param_tuple.append(k.get("stream", None))
            _param_tuple.append(k.get("reaction", None))
            _param_tuple.append(k.get("species", None))
            param_dict[tuple(_param_tuple)] = v
        return param_dict

    def operation_param_names(self):
        return list(self.name_to_ind.keys())

    def structure_params(self):
        return self._param_dict2list(self._structure_params)

    def physics_params(self):
        return self._param_dict2list(self._physics_params)

    def reaction_params(self):
        return self._param_dict2list(self._reaction_params)

    def transport_params(self):
        return self._param_dict2list(self._transport_params)

    def operation_params(self):
        return self._param_dict2list(self._operation_params)

    def operation_name2ind(self):
        param_list = []
        for name, param_tuple in self._operation_name2ind.items():
            _param_dict = {}
            _param_dict["param"] = param_tuple[0]
            if param_tuple[1] and param_tuple[1] in self.solids:
                _param_dict["solid"] = param_tuple[1]
            if param_tuple[1] and param_tuple[1] in self.gases:
                _param_dict["gas"] = param_tuple[1]
            if param_tuple[2] and param_tuple[2] in self.streams:
                _param_dict["stream"] = param_tuple[2]
            if param_tuple[3] and param_tuple[3] in self.reactions:
                _param_dict["reaction"] = param_tuple[3]
            if param_tuple[4] and param_tuple[4] in self.species:
                _param_dict["species"] = param_tuple[4]
            param_list.append((name, _param_dict))
        return param_list

    def measure_ind2name(self):
        param_list = []
        for param_tuple, name in self._measure_ind2name.items():
            _param_dict = {}
            _param_dict["param"] = param_tuple[0]
            if param_tuple[1] and param_tuple[1] in self.solids:
                _param_dict["solid"] = param_tuple[1]
            if param_tuple[1] and param_tuple[1] in self.gases:
                _param_dict["gas"] = param_tuple[1]
            if param_tuple[2] and param_tuple[2] in self.streams:
                _param_dict["stream"] = param_tuple[2]
            if param_tuple[3] and param_tuple[3] in self.reactions:
                _param_dict["reaction"] = param_tuple[3]
            if param_tuple[4] and param_tuple[4] in self.species:
                _param_dict["species"] = param_tuple[4]
            param_list.append((_param_dict, name))
        return param_list

    def var2unit(self):
        return self._var2unit
    
    def params(self):
        params = {}
        params.update(self._structure_params)
        params.update(self._physics_params)
        params.update(self._reaction_params)
        params.update(self._transport_params)
        params.update(self._operation_params)
        return self._param_dict2list(params)
