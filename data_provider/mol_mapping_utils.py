import re
import selfies as sf
import copy
import numpy as np
from rdkit import Chem
import random
from data_provider import sf_encoder
import warnings

SMI_REGEX_PATTERN = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#||\+|\\\\\/|:||@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
# SMI_REGEX_PATTERN = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#||\+|\\\\\/|:||@|\?|>|\*|\$|\%[0-9]{2}|[0-9]|\/|\\)"
SMI_REGEX = re.compile(SMI_REGEX_PATTERN)


def split_smiles(smiles, filter_=False):
    if False:
        smiles_tokens = [token for token in SMI_REGEX.findall(smiles) if token]
        start_pos = [0]
        for t in smiles_tokens[:-1]:
            start_pos.append(start_pos[-1] + len(t))
    else:
        # smiles_tokens_with_positions = [(match.group(), match.start()) for match in SMI_REGEX.finditer(smiles)]
        smiles_tokens = []
        start_pos = []
        for match in SMI_REGEX.finditer(smiles):
            result = match.group()
            if result:
                smiles_tokens.append(result)
                start_pos.append(match.start())

    if filter_:
        new_smiles_tokens = []
        new_start_pos = []
        for t, pos in zip(smiles_tokens, start_pos):
            if t in {
                "=",
                "(",
                ")",
                "#",
                "0",
                "1",
                "2",
                "3",
                "4",
                "5",
                "6",
                "7",
                "8",
                "9",
            }:
                continue
            if t.startswith("%"):
                continue
            new_smiles_tokens.append(t)
            new_start_pos.append(pos)
        return new_smiles_tokens, new_start_pos
    else:
        return smiles_tokens, start_pos


def sf_encode_and_attribute(smiles):
    selfies, attribution, my_attribution = sf_encoder.encoder(smiles, attribute=True)
    selfies_tokens = list(sf.split_selfies(selfies))
    attribution_tokens = [attr.token for attr in my_attribution]
    assert selfies_tokens == attribution_tokens
    ## in my_attribution, I use the index of selfies's original implementation, which is incorrect
    ## I am correcting it here
    for i in range(len(my_attribution)):
        my_attribution[i].index = i
    return selfies, my_attribution, selfies_tokens


def get_smiles2selfies_mapping(cano_smiles):
    selfies, attribution, selfies_tokens = sf_encode_and_attribute(cano_smiles)
    smiles_tokens, start_poses = split_smiles(cano_smiles, True)
    ## I already have rdmol_wh2smiles; now I need the mapping from smiles to selfies; I can build the mapping from selfies to smiles and then reverse it
    ## construct the correct mapping from smiles atom index to atom symbols

    atom_set = set()
    for attr in attribution:
        if attr.attribution is None:
            continue
        for item in attr.attribution:
            atom_set.add((item.index, item.token))
    atom_list = list(atom_set)
    atom_list.sort(key=lambda x: x[0])
    atom_mapping = {atom: i for i, atom in enumerate(atom_list)}

    selfies2smiles = []
    for attr in attribution:
        selfies2smiles.append([])
        if attr.attribution is None:
            continue
        for item in attr.attribution:
            atom_index = atom_mapping[(item.index, item.token)]
            sp = start_poses[atom_index]
            for i in range(len(item.token)):
                if item.token[i].isalpha():
                    selfies2smiles[-1].append(sp + i)

    ## have a sanity check here
    next_skip = False
    for i in range(len(selfies2smiles)):
        if (
            attribution[i].token.find("Ring") >= 0
            or attribution[i].token.find("Branch") >= 0
        ):
            next_skip = True
            continue
        if next_skip:
            next_skip = False
            continue
        sub_smiles = [cano_smiles[j] for j in selfies2smiles[i]]
        sub_smiles = "".join(sub_smiles)
        sub_sf_token = re.sub(r"[\d\[\]+\-\=\@\#\/\\]", "", attribution[i].token)

        ## sanity check
        if sub_sf_token.upper() == sub_smiles.upper():
            pass
        elif len(sub_sf_token) == 2 and sub_sf_token[-1] == "H":
            assert (
                sub_sf_token[0].upper() == sub_smiles.upper()
            ), f"{sub_sf_token} != {sub_smiles}\n{cano_smiles}\n{smiles_tokens}"
        # else:
        # warnings.warn(f'sub_sf_token != sub_smiles\n{sub_sf_token=}\n{sub_smiles=}\n{cano_smiles=}\n{smiles_tokens=}\n{selfies=}\n{selfies2smiles=}')

    ## construct the mapping from smiles to selfies
    smiles2selfies = {}
    for i, smiles_id_list in enumerate(selfies2smiles):
        for smiles_id in smiles_id_list:
            if smiles_id not in smiles2selfies:
                smiles2selfies[smiles_id] = [i]
            else:
                smiles2selfies[smiles_id].append(i)
    return smiles2selfies, selfies_tokens, selfies


############## for rdkit mol to smiles without H ##############

filter_regex = re.compile(r"[\(\)\[\]\{\}=\d\#\-+@%\/\\]")
two_letter_atoms = {
    "Al": "A",
    "Si": "Y",
    "Cl": "L",
    "As": "X",
    "Br": "R",
    "Hg": "G",
    "Bi": "B",
}


def obtain_atoms_from_smiles(text, regex):
    # Replace patterns with an empty string
    output = regex.sub("", text).upper()
    # Replace two-letter atoms with a single letter
    for atom, replacement in two_letter_atoms.items():
        output = output.replace(atom.upper(), replacement)
    ## obtain the mapping from the output text to the input text
    masked_text = regex.sub(
        "*", text
    ).upper()  # there is a caveat: for atoms with two letters, both letters are treated as one atom
    ## deal with Cl and Br
    masked_text = re.sub(r"AL|SI|CL|AS|BR|HG|BI", "X*", masked_text)
    masked_text = np.asarray(list(masked_text))
    assert len(masked_text) == len(text)
    mapping = np.where(masked_text != "*")[0]
    assert len(mapping) == len(output)
    return output, mapping


invalid_int = -99999


def build_rdkit2cano_smiles_withoutH_mapping(rdmol):
    """
    In the end, I need the mapping between rdkit with h to smiles without h
    """
    rdmol = copy.deepcopy(rdmol)
    for atom in rdmol.GetAtoms():
        atom.SetProp("atom_index", str(atom.GetIdx()))
    rdmol_woh = Chem.RemoveHs(rdmol)
    canonical_smiles = Chem.MolToSmiles(rdmol_woh, canonical=True)
    smiles_atom_order = rdmol_woh.GetPropsAsDict(True, True)["_smilesAtomOutputOrder"]
    rdmol_woh = Chem.RenumberAtoms(rdmol_woh, list(smiles_atom_order))

    rdmol_wh2rdmol_woh = np.full(
        rdmol.GetNumAtoms(), invalid_int
    )  # fill the array with invalid int
    # build the mapping from rdmol with h to rdmol without h
    for i, atom in enumerate(rdmol_woh.GetAtoms()):
        assert i == atom.GetIdx()
        rdmol_wh2rdmol_woh[int(atom.GetProp("atom_index"))] = atom.GetIdx()

    # atoms_in_rdmol_woh = ''.join([atom.GetSymbol() for atom in rdmol_woh.GetAtoms()]).upper()
    symbols = []
    for atom in rdmol_woh.GetAtoms():
        symbol = atom.GetSymbol()
        if len(symbol) == 2:
            symbols.append(two_letter_atoms[symbol])
        else:
            symbols.append(symbol)
    atoms_in_rdmol_woh = "".join(symbols).upper()
    atoms_in_smiles, output2input = obtain_atoms_from_smiles(
        canonical_smiles, filter_regex
    )
    if atoms_in_rdmol_woh == atoms_in_smiles:
        rdmol_woh2smiles = output2input
    else:
        # this is to fix the problem that canonical smiles has more H than the rdmol_woh
        assert len(atoms_in_rdmol_woh) < len(atoms_in_smiles)
        add_H = 0
        rdmol_woh2smiles = []
        for j in range(len(atoms_in_smiles)):
            if j == len(atoms_in_smiles) - 1 and (j - add_H) == len(atoms_in_rdmol_woh):
                assert atoms_in_smiles[j] == "H", print(
                    atoms_in_smiles,
                    atoms_in_rdmol_woh,
                    len(atoms_in_smiles),
                    len(atoms_in_rdmol_woh),
                    rdmol.GetNumAtoms(),
                )
                continue
            if atoms_in_smiles[j].upper() != atoms_in_rdmol_woh[j - add_H].upper():
                assert atoms_in_smiles[j] == "H", print(
                    atoms_in_smiles,
                    atoms_in_rdmol_woh,
                    len(atoms_in_smiles),
                    len(atoms_in_rdmol_woh),
                    rdmol.GetNumAtoms(),
                    atoms_in_smiles[j],
                    atoms_in_rdmol_woh[j - add_H],
                )
                add_H += 1
            else:
                rdmol_woh2smiles.append(output2input[j])
        rdmol_woh2smiles = np.asarray(rdmol_woh2smiles)
        assert len(rdmol_woh2smiles) == len(atoms_in_rdmol_woh)

    # TODO: comment the following code because it is an incorrect assertion then Cl, Br or Bi is in the smiles
    # indexed_smiles = ''.join([canonical_smiles[i] for i in rdmol_woh2smiles]).upper()
    # assert atoms_in_rdmol_woh == indexed_smiles, print(atoms_in_rdmol_woh, indexed_smiles)

    rdmol_wh2smiles = []
    for i, j in enumerate(rdmol_wh2rdmol_woh):
        j = int(j)
        if j == invalid_int:
            rdmol_wh2smiles.append(invalid_int)
        else:
            ## sanity check
            sym = rdmol.GetAtomWithIdx(i).GetSymbol()
            if len(sym) == 2:
                assert (
                    sym.upper()
                    == canonical_smiles[
                        rdmol_woh2smiles[j] : rdmol_woh2smiles[j] + 2
                    ].upper()
                ), print(
                    rdmol.GetAtomWithIdx(i).GetSymbol(),
                    canonical_smiles[rdmol_woh2smiles[j]].upper(),
                )
            elif len(sym) == 1:
                assert sym == canonical_smiles[rdmol_woh2smiles[j]].upper(), print(
                    sym, canonical_smiles[rdmol_woh2smiles[j]].upper(), canonical_smiles
                )
            else:
                raise NotImplementedError()
            rdmol_wh2smiles.append(rdmol_woh2smiles[j])
    rdmol_wh2smiles = np.asarray(rdmol_wh2smiles)
    return rdmol_wh2smiles, canonical_smiles


## this code is deprecated
def build_rdkit2cano_smiles_withH_mapping(mol):
    num_atoms = mol.GetNumAtoms()
    conf_pos = mol.GetConformer().GetPositions()
    assert num_atoms == conf_pos.shape[0]
    canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
    smiles_atom_order = mol.GetPropsAsDict(True, True)["_smilesAtomOutputOrder"]
    mol_canonical = Chem.RenumberAtoms(mol, list(smiles_atom_order))
    atoms_in_rdkit = "".join([atom.GetSymbol() for atom in mol_canonical.GetAtoms()])
    atoms_in_smiles, output2input = obtain_atoms_from_smiles(
        canonical_smiles, filter_regex
    )

    assert len(atoms_in_rdkit) == num_atoms
    ## I need to build a mapping from the atoms in RDKit to the atoms in the SMILES
    if len(atoms_in_smiles) == len(atoms_in_rdkit):
        rdkit2smiles = output2input
    else:
        assert len(atoms_in_smiles) > len(atoms_in_rdkit)
        add_H = 0
        rdkit2smiles = []
        for j in range(len(atoms_in_smiles)):
            if j == len(atoms_in_smiles) - 1 and (j - add_H) == len(atoms_in_rdkit):
                assert atoms_in_smiles[j] == "H", print(
                    atoms_in_smiles,
                    atoms_in_rdkit,
                    len(atoms_in_smiles),
                    len(atoms_in_rdkit),
                    num_atoms,
                )
                continue
            if atoms_in_smiles[j] != atoms_in_rdkit[j - add_H]:
                assert atoms_in_smiles[j] == "H", print(
                    atoms_in_smiles,
                    atoms_in_rdkit,
                    len(atoms_in_smiles),
                    len(atoms_in_rdkit),
                    num_atoms,
                )
                add_H += 1
            else:
                rdkit2smiles.append(output2input[j])
        rdkit2smiles = np.asarray(rdkit2smiles)
        assert len(rdkit2smiles) == len(atoms_in_rdkit)
    ## check if the atoms are in the same order
    assert atoms_in_rdkit == "".join(
        [canonical_smiles[i] for i in rdkit2smiles]
    ), print(atoms_in_rdkit, "".join([canonical_smiles[i] for i in rdkit2smiles]))
    return mol_canonical, rdkit2smiles, canonical_smiles


def fix_molecule_valence(rdmol):
    """
    自动检测并修复分子中的化合价异常问题
    """
    # 创建分子的深拷贝以避免修改原始分子
    mol = copy.deepcopy(rdmol)

    # 不严格更新属性缓存
    mol.UpdatePropertyCache(strict=False)

    # 检测并修复每个原子的化合价问题
    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        symbol = atom.GetSymbol()
        formal_charge = atom.GetFormalCharge()
        explicit_valence = atom.GetExplicitValence()

        # 根据元素类型和形式电荷，计算理论上的最大允许化合价
        max_valence = {
            "C": 4,
            "N": 5,
            "O": 2,
            "S": 6,
            "P": 5,
            "F": 1,
            "Cl": 1,
            "Br": 1,
            "I": 1,
            "H": 1,
            "B": 3,
            "Si": 4,
        }.get(
            symbol, 4
        )  # 默认值设为4，适用于大多数常见元素

        # 考虑形式电荷对允许化合价的影响
        adjusted_max_valence = max_valence + formal_charge

        # 检查是否存在化合价异常
        if explicit_valence > adjusted_max_valence:
            # 计算需要添加的显式氢原子数量
            needed_hs = explicit_valence - adjusted_max_valence
            # 设置显式氢原子数量
            atom.SetNumExplicitHs(atom.GetNumExplicitHs() + needed_hs)
            # 更新原子的属性
            atom.UpdatePropertyCache(strict=False)

    # 尝试进行清理操作，但不进行严格检查
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ADJUSTHS)
    except Exception:
        # 如果清理失败，至少我们已经尝试调整了氢原子数量
        print(f'Failed to sanitize molecule {rdmol.GetProp("smiles")}')
        pass

    return mol


def build_rdkit2rand_smiles_withoutH_mapping(rdmol, rand_smiles=None, addHs=False):
    """
    In the end, I need the mapping between rdkit with h to smiles without h
    """
    rdmol = copy.deepcopy(rdmol)
    for atom in rdmol.GetAtoms():
        atom.SetProp("atom_index", str(atom.GetIdx()))
    if not addHs:
        try:
            rdmol_woh = Chem.RemoveHs(rdmol)
        except Chem.rdchem.AtomValenceException as e:
            print(e)
            fixed_rdmol_woh = fix_molecule_valence(rdmol)
            rdmol_woh = fixed_rdmol_woh
    else:
        rdmol_woh = rdmol

    if rand_smiles == "restricted":
        random_order = list(range(rdmol_woh.GetNumAtoms()))
        random.shuffle(random_order)
        random_mol = Chem.RenumberAtoms(rdmol_woh, newOrder=random_order)
        output_smiles = Chem.MolToSmiles(
            random_mol, canonical=False, isomericSmiles=False
        )
        smiles_atom_order = random_mol.GetPropsAsDict(True, True)[
            "_smilesAtomOutputOrder"
        ]
        rdmol_woh = Chem.RenumberAtoms(random_mol, list(smiles_atom_order))
    elif rand_smiles in {"None", "none", "False", "false"} or (not rand_smiles):
        output_smiles = Chem.MolToSmiles(rdmol_woh, canonical=False)
        smiles_atom_order = rdmol_woh.GetPropsAsDict(True, True)[
            "_smilesAtomOutputOrder"
        ]
        rdmol_woh = Chem.RenumberAtoms(rdmol_woh, list(smiles_atom_order))
    elif rand_smiles == "canonical":
        output_smiles = Chem.MolToSmiles(rdmol_woh, canonical=True)
        smiles_atom_order = rdmol_woh.GetPropsAsDict(True, True)[
            "_smilesAtomOutputOrder"
        ]
        rdmol_woh = Chem.RenumberAtoms(rdmol_woh, list(smiles_atom_order))
    else:
        raise NotImplementedError()

    rdmol_wh2rdmol_woh = np.full(
        rdmol.GetNumAtoms(), invalid_int
    )  # fill the array with invalid int
    # build the mapping from rdmol with h to rdmol without h
    for i, atom in enumerate(rdmol_woh.GetAtoms()):
        assert i == atom.GetIdx()
        rdmol_wh2rdmol_woh[int(atom.GetProp("atom_index"))] = atom.GetIdx()

    # atoms_in_rdmol_woh = ''.join([atom.GetSymbol() for atom in rdmol_woh.GetAtoms()]).upper()
    symbols = []
    for atom in rdmol_woh.GetAtoms():
        symbol = atom.GetSymbol()
        if len(symbol) == 2:
            symbols.append(two_letter_atoms[symbol])
        else:
            symbols.append(symbol)
    atoms_in_rdmol_woh = "".join(symbols).upper()
    atoms_in_smiles, output2input = obtain_atoms_from_smiles(
        output_smiles, filter_regex
    )
    if atoms_in_rdmol_woh == atoms_in_smiles:
        rdmol_woh2smiles = output2input
    else:
        # this is to fix the problem that the output smiles has more H than the rdmol_woh
        assert len(atoms_in_rdmol_woh) < len(atoms_in_smiles)
        add_H = 0
        rdmol_woh2smiles = []
        for j in range(len(atoms_in_smiles)):
            if j == len(atoms_in_smiles) - 1 and (j - add_H) == len(atoms_in_rdmol_woh):
                assert atoms_in_smiles[j] == "H", print(
                    atoms_in_smiles,
                    atoms_in_rdmol_woh,
                    len(atoms_in_smiles),
                    len(atoms_in_rdmol_woh),
                    rdmol.GetNumAtoms(),
                )
                continue
            if atoms_in_smiles[j].upper() != atoms_in_rdmol_woh[j - add_H].upper():
                assert atoms_in_smiles[j] == "H", print(
                    atoms_in_smiles,
                    atoms_in_rdmol_woh,
                    len(atoms_in_smiles),
                    len(atoms_in_rdmol_woh),
                    rdmol.GetNumAtoms(),
                    atoms_in_smiles[j],
                    atoms_in_rdmol_woh[j - add_H],
                )
                add_H += 1
            else:
                rdmol_woh2smiles.append(output2input[j])
        rdmol_woh2smiles = np.asarray(rdmol_woh2smiles)
        assert len(rdmol_woh2smiles) == len(atoms_in_rdmol_woh)

    # TODO: comment the following code because it is an incorrect assertion then Cl, Br or Bi is in the smiles
    # indexed_smiles = ''.join([canonical_smiles[i] for i in rdmol_woh2smiles]).upper()
    # assert atoms_in_rdmol_woh == indexed_smiles, print(atoms_in_rdmol_woh, indexed_smiles)

    rdmol_wh2smiles = []
    for i, j in enumerate(rdmol_wh2rdmol_woh):
        j = int(j)
        if j == invalid_int:
            rdmol_wh2smiles.append(invalid_int)
        else:
            ## sanity check
            sym = rdmol.GetAtomWithIdx(i).GetSymbol()
            if len(sym) == 2:
                assert (
                    sym.upper()
                    == output_smiles[
                        rdmol_woh2smiles[j] : rdmol_woh2smiles[j] + 2
                    ].upper()
                ), print(
                    rdmol.GetAtomWithIdx(i).GetSymbol(),
                    output_smiles[rdmol_woh2smiles[j]].upper(),
                )
            elif len(sym) == 1:
                assert sym == output_smiles[rdmol_woh2smiles[j]].upper(), print(
                    sym, output_smiles[rdmol_woh2smiles[j]].upper(), output_smiles
                )
            else:
                raise NotImplementedError()
            rdmol_wh2smiles.append(rdmol_woh2smiles[j])
    rdmol_wh2smiles = np.asarray(rdmol_wh2smiles)
    return rdmol_wh2smiles, output_smiles
