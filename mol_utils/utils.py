import torch
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import selfies as sf
import re
import warnings
from typing import List, Union, Tuple

from selfies.compatibility import modernize_symbol
from selfies.exceptions import DecoderError
from selfies.grammar_rules import (
    get_index_from_selfies,
    next_atom_state,
    next_branch_state,
    next_ring_state,
    process_atom_symbol,
    process_branch_symbol,
    process_ring_symbol
)
from selfies.mol_graph import MolecularGraph, Attribution
from selfies.utils.selfies_utils import split_selfies
from selfies.utils.smiles_utils import mol_to_smiles

def smiles2conformations(smiles, seed, filter_h_only=True, optimizer='MMFF', number_of_conformations=5):

    results = []
    if optimizer == 'MMFF':
        optimize = AllChem.MMFFOptimizeMolecule
    elif optimizer == 'UFF':
        optimize = AllChem.UFFOptimizeMolecule
    else:
        raise ValueError('optimize_algo should be either MMFF or UFF')

    for _ in range(number_of_conformations):
        molecule = Chem.MolFromSmiles(smiles)
        molecule = AllChem.AddHs(molecule)
        atoms = [atom.GetSymbol() for atom in molecule.GetAtoms()]

        if filter_h_only:
            if (np.asarray(atoms) == 'H').all():
                results.append(None)
                continue

        coordinate_list = []

        embed_result = AllChem.EmbedMolecule(molecule, maxAttempts=1000, randomSeed=seed)
        if embed_result == -1: # Embedding failed, try again
            molecule = Chem.MolFromSmiles(smiles)
            AllChem.EmbedMolecule(molecule, maxAttempts=1000, randomSeed=seed)
            molecule = AllChem.AddHs(molecule, addCoords=True)
            atoms = [atom.GetSymbol() for atom in molecule.GetAtoms()]

        try:
            optimize(molecule)
        except:
            pass

        coordinates = molecule.GetConformer().GetPositions()

        assert len(atoms) == len(coordinates), "coordinates shape is not align with {}".format(smiles)

        coordinate_list.append(coordinates.astype(np.float32))
        results.append({'atoms': atoms, 'coordinates': coordinate_list})

    return results

def selfies2smiles(selfies, embedding_selfies=None, normalize=True):
    assert isinstance(selfies, str), "selfies should be a string"

    assert (sf.len_selfies(selfies) == embedding_selfies.shape[0]) or (embedding_selfies is None), "embedding shape is not align with selfies"

    smiles, attribution = modified_decoder(selfies, attribute=True)

    atom_finder = re.compile(r"""
    (
    Cl? |             # Cl and Br are part of the organic subset
    Br? |
    [NOSPFIbcnosp*] | # as are these single-letter elements
    \[[^]]+\]         # everything else must be in []s
    )
    """, re.X)

    positions = [(x.start(), x.end()) for x in atom_finder.finditer(smiles)]

    molecule = Chem.MolFromSmiles(smiles)

    embedding_smiles = torch.zeros(molecule.GetNumAtoms())

    for smiles_token in attribution:

        for i, (start, end) in enumerate(positions):
            if start <= smiles_token.index <= end:
                atom_index = i
                # break

        for selfies_token in smiles_token.attribution:
            embedding_smiles[atom_index] += embedding_selfies[selfies_token.index]

    if normalize:
        embedding_smiles = embedding_smiles / embedding_smiles.sum()

    return smiles, embedding_smiles

def smiles2selfies(smiles, embedding_smiles=None, normalize=True):
    assert isinstance(smiles, str), "selfies should be a string"

    molecule = Chem.MolFromSmiles(smiles)

    assert (molecule.GetNumAtoms() == embedding_smiles.shape[0]) or (embedding_smiles is None), "embedding shape is not align with smiles"

    selfies, attribution = sf.encoder(smiles, attribute=True)

    embedding_selfies = torch.zeros(molecule.GetNumAtoms())

    for selfies_token in attribution:

        for i, atom in enumerate(molecule.GetAtoms()):
            if atom.GetSymbol() == selfies_token.tokens[1:-1]:
                atom_index = i
                # break

        for smiles_token in selfies_token.attribution:
            embedding_selfies[atom_index] += embedding_selfies[smiles_token.index]

    if normalize:
        embedding_selfies = embedding_selfies / embedding_selfies.sum()

    return selfies, embedding_selfies

def modified_decoder(
        selfies: str,
        compatible: bool = False,
        attribute: bool = False) ->\
        Union[str, Tuple[str, List[Tuple[str,  List[Tuple[int, str]]]]]]:
    """Translates a SELFIES string into its corresponding SMILES string.

    This translation is deterministic but depends on the current semantic
    constraints. The output SMILES string is guaranteed to be syntatically
    correct and guaranteed to represent a molecule that obeys the
    semantic constraints.

    :param selfies: the SELFIES string to be translated.
    :param compatible: if ``True``, this function will accept SELFIES strings
        containing depreciated symbols from previous releases. However, this
        function may behave differently than in previous major relases,
        and should not be treated as backard compatible.
        Defaults to ``False``.
    :param attribute: if ``True``, an attribution map connecting selfies
        tokens to smiles tokens is output.
    :return: a SMILES string derived from the input SELFIES string.
    :raises DecoderError: if the input SELFIES string is malformed.

    :Example:

    >>> import selfies as sf
    >>> modified_decoder('[C][=C][F]')
    'C=CF'
    """

    if compatible:
        msg = "\nselfies.decoder() may behave differently than in previous " \
              "major releases. We recommend using SELFIES that are up to date."
        warnings.warn(msg, stacklevel=2)

    mol = MolecularGraph(attributable=attribute)

    rings = []
    attribution_index = 0
    for s in selfies.split("."):
        n = _derive_mol_from_symbols(
            symbol_iter=enumerate(_tokenize_selfies(s, compatible)),
            mol=mol,
            selfies=selfies,
            max_derive=float("inf"),
            init_state=0,
            root_atom=None,
            rings=rings,
            attribute_stack=[] if attribute else None,
            attribution_index=attribution_index
        )
        attribution_index += n
    _form_rings_bilocally(mol, rings)
    return mol


def _tokenize_selfies(selfies, compatible):
    if isinstance(selfies, str):
        symbol_iter = split_selfies(selfies)
    elif isinstance(selfies, list):
        symbol_iter = selfies
    else:
        raise ValueError()  # should not happen

    try:
        for symbol in symbol_iter:
            if symbol == "[nop]":
                continue
            if compatible:
                symbol = modernize_symbol(symbol)
            yield symbol
    except ValueError as err:
        raise DecoderError(str(err)) from None


def _derive_mol_from_symbols(
        symbol_iter, mol, selfies, max_derive,
        init_state, root_atom, rings, attribute_stack, attribution_index
):
    n_derived = 0
    state = init_state
    prev_atom = root_atom

    while (state is not None) and (n_derived < max_derive):

        try:  # retrieve next symbol
            index, symbol = next(symbol_iter)
            n_derived += 1
        except StopIteration:
            break

        # Case 1: Branch symbol (e.g. [Branch1])
        if "ch" == symbol[-4:-2]:

            output = process_branch_symbol(symbol)
            if output is None:
                _raise_decoder_error(selfies, symbol)
            btype, n = output

            if state <= 1:
                next_state = state
            else:
                binit_state, next_state = next_branch_state(btype, state)

                Q = _read_index_from_selfies(symbol_iter, n_symbols=n)
                n_derived += n + _derive_mol_from_symbols(
                    symbol_iter, mol, selfies, (Q + 1),
                    init_state=binit_state, root_atom=prev_atom, rings=rings,
                    attribute_stack=attribute_stack +
                    [Attribution(index + attribution_index, symbol)
                     ] if attribute_stack is not None else None,
                    attribution_index=attribution_index
                )

        # Case 2: Ring symbol (e.g. [Ring2])
        elif "ng" == symbol[-4:-2]:

            output = process_ring_symbol(symbol)
            if output is None:
                _raise_decoder_error(selfies, symbol)
            ring_type, n, stereo = output

            if state == 0:
                next_state = state
            else:
                ring_order, next_state = next_ring_state(ring_type, state)
                bond_info = (ring_order, stereo)

                Q = _read_index_from_selfies(symbol_iter, n_symbols=n)
                # n_derived += n
                n_derived += n + _derive_mol_from_symbols(
                    symbol_iter, mol, selfies, (Q + 1),
                    init_state=binit_state, root_atom=prev_atom, rings=rings,
                    attribute_stack=attribute_stack +
                    [Attribution(index + attribution_index, symbol)
                     ] if attribute_stack is not None else None,
                    attribution_index=attribution_index
                )
                lidx = max(0, prev_atom.index - (Q + 1))
                rings.append((mol.get_atom(lidx), prev_atom, bond_info))

        # Case 3: [epsilon]
        elif "eps" in symbol:
            next_state = 0 if (state == 0) else None

        # Case 4: regular symbol (e.g. [N], [=C], [F])
        else:

            output = process_atom_symbol(symbol)
            if output is None:
                _raise_decoder_error(selfies, symbol)
            (bond_order, stereo), atom = output
            cap = atom.bonding_capacity

            bond_order, next_state = next_atom_state(bond_order, cap, state)
            if bond_order == 0:
                if state == 0:
                    o = mol.add_atom(atom, True)
                    mol.add_attribution(
                        o,  attribute_stack +
                        [Attribution(index + attribution_index, symbol)]
                        if attribute_stack is not None else None)
            else:
                o = mol.add_atom(atom)
                mol.add_attribution(
                    o, attribute_stack +
                    [Attribution(index + attribution_index, symbol)]
                    if attribute_stack is not None else None)
                src, dst = prev_atom.index, atom.index
                o = mol.add_bond(src=src, dst=dst,
                                 order=bond_order, stereo=stereo)
                mol.add_attribution(
                    o, attribute_stack +
                    [Attribution(index + attribution_index, symbol)]
                    if attribute_stack is not None else None)
            prev_atom = atom

        if next_state is None:
            break
        state = next_state

    while n_derived < max_derive:  # consume remaining tokens
        try:
            next(symbol_iter)
            n_derived += 1
        except StopIteration:
            break

    return n_derived


def _raise_decoder_error(selfies, invalid_symbol):
    err_msg = "invalid symbol '{}'\n\tSELFIES: {}".format(
        invalid_symbol, selfies
    )
    raise DecoderError(err_msg)


def _read_index_from_selfies(symbol_iter, n_symbols):
    index_symbols = []
    for _ in range(n_symbols):
        try:
            index_symbols.append(next(symbol_iter)[-1])
        except StopIteration:
            index_symbols.append(None)
    return get_index_from_selfies(*index_symbols)


def _form_rings_bilocally(mol, rings):
    rings_made = [0] * len(mol)

    for latom, ratom, bond_info in rings:
        lidx, ridx = latom.index, ratom.index

        if lidx == ridx:  # ring to the same atom forbidden
            continue

        order, (lstereo, rstereo) = bond_info
        lfree = latom.bonding_capacity - mol.get_bond_count(lidx)
        rfree = ratom.bonding_capacity - mol.get_bond_count(ridx)

        if lfree <= 0 or rfree <= 0:
            continue  # no room for ring bond
        order = min(order, lfree, rfree)

        if mol.has_bond(a=lidx, b=ridx):
            bond = mol.get_dirbond(src=lidx, dst=ridx)
            new_order = min(order + bond.order, 3)
            mol.update_bond_order(a=lidx, b=ridx, new_order=new_order)

        else:
            mol.add_ring_bond(
                a=lidx, a_stereo=lstereo, a_pos=rings_made[lidx],
                b=ridx, b_stereo=rstereo, b_pos=rings_made[ridx],
                order=order
            )
            rings_made[lidx] += 1
            rings_made[ridx] += 1
