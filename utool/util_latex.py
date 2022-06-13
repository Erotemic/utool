# -*- coding: utf-8 -*-
"""
TODO: box and whisker
http://tex.stackexchange.com/questions/115210/boxplot-in-latex
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from six.moves import range, map, zip
import os
import re
import textwrap
try:
    import numpy as np
except ImportError:
    pass
from os.path import join, splitext, dirname  # NOQA
from utool import util_num
from utool import util_inject
print, rrr, profile = util_inject.inject2(__name__)

#def ensure_latex_environ():
#    paths = os.environ['PATH'].split(os.pathsep)
#    mpl.rc('font',**{'family':'serif'})
#    mpl.rc('text', usetex=True)
#    mpl.rc('text.latex',unicode=True)
#    mpl.rc('text.latex',preamble='\usepackage[utf8]{inputenc}')


def find_ghostscript_exe():
    import utool as ut
    if ut.WIN32:
        gs_exe = r'C:\Program Files (x86)\gs\gs9.16\bin\gswin32c.exe'
    else:
        gs_exe = 'gs'
    return gs_exe


def compress_pdf(pdf_fpath, output_fname=None):
    """ uses ghostscript to write a pdf """
    import utool as ut
    ut.assertpath(pdf_fpath)
    suffix = '_' + ut.get_datestamp(False) + '_compressed'
    print('pdf_fpath = %r' % (pdf_fpath,))
    output_pdf_fpath = ut.augpath(pdf_fpath, suffix, newfname=output_fname)
    print('output_pdf_fpath = %r' % (output_pdf_fpath,))
    gs_exe = find_ghostscript_exe()
    cmd_list = (
        gs_exe,
        '-sDEVICE=pdfwrite',
        '-dCompatibilityLevel=1.4',
        '-dNOPAUSE',
        '-dQUIET',
        '-dBATCH',
        '-sOutputFile=' + output_pdf_fpath,
        pdf_fpath
    )
    ut.cmd(*cmd_list)
    return output_pdf_fpath


def make_full_document(text, title=None, preamp_decl={}, preamb_extra=None):
    r"""
    dummy preamble and document to wrap around latex fragment

    Args:
        text (str):
        title (str):

    Returns:
        str: text_

    CommandLine:
        python -m utool.util_latex --test-make_full_document

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_latex import *  # NOQA
        >>> text = 'foo'
        >>> title = 'title'
        >>> preamp_decl = {}
        >>> text_ = make_full_document(text, title)
        >>> result = str(text_)
        >>> print(result)
    """
    import utool as ut
    doc_preamb = ut.codeblock('''
    %\\documentclass{article}
    \\documentclass[10pt,twocolumn,letterpaper]{article}
    % \\usepackage[utf8]{inputenc}
    \\usepackage[T1]{fontenc}

    \\usepackage{times}
    \\usepackage{epsfig}
    \\usepackage{graphicx}
    \\usepackage{amsmath,amsthm,amssymb}
    \\usepackage[usenames,dvipsnames,svgnames,table]{xcolor}
    \\usepackage{multirow}
    \\usepackage{subcaption}
    \\usepackage{booktabs}

    %\\pagenumbering{gobble}
    ''')
    if preamb_extra is not None:
        if isinstance(preamb_extra, (list, tuple)):
            preamb_extra = '\n'.join(preamb_extra)
        doc_preamb += '\n' +  preamb_extra + '\n'
    if title is not None:
        preamp_decl['title'] = title

    decl_lines = [r'\{key}{{{val}}}'.format(key=key, val=val)
                  for key, val in preamp_decl.items()]
    doc_decllines = '\n'.join(decl_lines)

    doc_header = ut.codeblock(
        r'''
        \begin{document}
        ''')
    if preamp_decl.get('title') is not None:
        doc_header += r'\maketitle'

    doc_footer = ut.codeblock(
        r'''
        \end{document}
        ''')
    text_ = '\n'.join((doc_preamb, doc_decllines, doc_header, text, doc_footer))
    return text_


def render_latex_text(input_text, nest_in_doc=False, preamb_extra=None,
                      appname='utool', verbose=None):
    """ compiles latex and shows the result """
    import utool as ut
    if verbose is None:
        verbose = ut.VERBOSE
    dpath = ut.ensure_app_resource_dir(appname, 'latex_tmp')
    # put a latex framgent in a full document
    # print(input_text)
    fname = 'temp_render_latex'
    pdf_fpath = ut.compile_latex_text(
        input_text, dpath=dpath, fname=fname, preamb_extra=preamb_extra,
        verbose=verbose)
    ut.startfile(pdf_fpath)
    return pdf_fpath


def compile_latex_text(input_text, dpath=None, fname=None, verbose=True,
                       move=True, nest_in_doc=None, title=None,
                       preamb_extra=None):
    r"""
    CommandLine:
        python -m utool.util_latex --test-compile_latex_text --show

    Ignore:
        pdflatex -shell-escape --synctex=-1 -src-specials -interaction=nonstopmode\
            ~/code/ibeis/tmptex/latex_formatter_temp.tex

    Example1:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_latex import *  # NOQA
        >>> import utool as ut
        >>> verbose = True
        >>> #dpath = '/home/joncrall/code/ibeis/aidchallenge'
        >>> dpath = dirname(ut.grab_test_imgpath())
        >>> #ut.vd(dpath)
        >>> orig_fpaths = ut.list_images(dpath, fullpath=True)
        >>> figure_str = ut.get_latex_figure_str(orig_fpaths, width_str='2.4in', nCols=2)
        >>> input_text = figure_str
        >>> pdf_fpath = ut.compile_latex_text(input_text, dpath=dpath,
        >>>                                   verbose=verbose)
        >>> output_pdf_fpath = ut.compress_pdf(pdf_fpath)
        >>> print(pdf_fpath)
        >>> ut.quit_if_noshow()
        >>> ut.startfile(pdf_fpath)
    """
    import utool as ut
    if verbose:
        print('[ut] compile_latex_text')

    if nest_in_doc is None:
        nest_in_doc = 'documentclass' not in input_text
    if nest_in_doc:
        text = make_full_document(input_text, title=title,
                                  preamb_extra=preamb_extra)
    if not dpath:
        dpath = os.getcwd()
    if fname is None:
        fname = 'temp_latex'

    # Create temporary work directly
    work_dpath = join(dpath, '.tmptex')
    ut.ensuredir(work_dpath, verbose=verbose > 1)

    fname_tex = ut.ensure_ext(fname, '.tex')
    fname_pdf = ut.ensure_ext(fname, '.pdf')

    tex_fpath = join(work_dpath, fname_tex)
    pdf_fpath_output = join(work_dpath, fname_pdf)
    ut.write_to(tex_fpath, text)

    with ut.ChdirContext(work_dpath, verbose=verbose > 1):
        # print(text)
        args = ' '.join([
            'lualatex', '-shell-escape', '--synctex=-1', '-src-specials',
            '-interaction=nonstopmode', tex_fpath
        ])
        info = ut.cmd2(args, verbose=verbose > 1)
        if not ut.checkpath(pdf_fpath_output, verbose=verbose > 1):
            print('Error compiling LaTeX')
            ut.print_code(text, 'latex')
            print(info['out'])
            raise RuntimeError('latex failed ')

    if move:
        pdf_fpath = join(dpath, fname_pdf)
        ut.move(pdf_fpath_output, pdf_fpath, verbose=verbose > 1)
    else:
        pdf_fpath = pdf_fpath_output
    return pdf_fpath


def convert_pdf_to_image(pdf_fpath, ext='.jpg', verbose=1, dpi=300,
                         quality=90):
    import utool as ut
    if verbose:
        print('[ut] convert_pdf_to_image.')
    img_fpath = ut.ensure_ext(pdf_fpath, ext)
    if ut.UNIX:
        convert_fpath = ut.cmd2('which convert')['out'].strip()
        if not convert_fpath:
            raise Exception('ImageMagik convert was not found')
    args = ' '.join(['convert', '-density', str(dpi), pdf_fpath, '-quality',
                     str(quality), img_fpath])
    info = ut.cmd2(args, verbose=verbose > 1)  # NOQA
    if not ut.checkpath(img_fpath, verbose=verbose > 1):
        print('Failed to convert pdf to ' + ext)
        print(info['out'])
        raise Exception('ImageMagik failed to convert pdf to ' + ext)
    return img_fpath


def render_latex(input_text, dpath=None, fname=None, preamb_extra=None,
                 verbose=1, **kwargs):
    """
    Renders latex text into a jpeg.

    Whitespace that would have appeared in the PDF is removed, so the jpeg is
    cropped only the the relevant part.  This is ideal for figures that only
    take a single page.

    Args:
        input_text (?):
        dpath (str):  directory path(default = None)
        fname (str):  file name(default = None)
        preamb_extra (None): (default = None)
        verbose (int):  verbosity flag(default = 1)

    Returns:
        str: jpg_fpath -  file path string

    CommandLine:
        python -m utool.util_latex render_latex '$O(n^2)$' --fpath=~/slides/tmp.jpg

    Script:
        >>> # SCRIPT
        >>> from utool.util_latex import *  # NOQA
        >>> from os.path import split, expanduser
        >>> import utool as ut
        >>> input_text = ' '.join(ut.get_varargs()[1:])
        >>> dpath, fname = split(ut.argval('--fpath', ''))
        >>> dpath = expanduser(ut.argval('--dpath', dpath))
        >>> fname = ut.argval('--fname', fname)
        >>> kwargs = ut.dict_subset(ut.argparse_funckw(ut.convert_pdf_to_image), ['dpi', 'quality'])
        >>> jpg_fpath = render_latex(input_text, dpath, fname, **kwargs)
        >>> if ut.argflag('--diskshow'):
        >>>     ut.startfile(jpg_fpath)
    """
    import utool as ut
    try:
        import vtool_ibeis as vt
    except ImportError:
        import vtool as vt
    # turn off page numbers
    input_text_ = '\\pagenumbering{gobble}\n' + input_text
    # fname, _ = splitext(fname)
    img_fname = ut.ensure_ext(fname, ['.jpg'] + list(ut.IMG_EXTENSIONS))
    img_fpath = join(dpath, img_fname)
    pdf_fpath = ut.compile_latex_text(
        input_text_, fname=fname, dpath=dpath, preamb_extra=preamb_extra,
        verbose=verbose, move=False)
    ext = splitext(img_fname)[1]
    fpath_in = ut.convert_pdf_to_image(pdf_fpath, ext=ext, verbose=verbose)
    # Clip of boundaries of the pdf imag
    vt.clipwhite_ondisk(fpath_in, fpath_out=img_fpath, verbose=verbose > 1)
    return img_fpath


def latex_multicolumn(data, ncol=2, alignstr='|c|'):
    data = escape_latex(data)
    return r'\multicolumn{%d}{%s}{%s}' % (ncol, alignstr, data)


def latex_multirow(data, nrow=2):
    return r'\multirow{%d}{*}{%s}' % (nrow, data)


def latex_get_stats(lbl, data, mode=0):
    import utool as ut
    stats_ = ut.get_stats(data)
    if stats_.get('empty_list', False):
        return '% NA: latex_get_stats, data=[]'
    try:
        max_ = stats_['max']
        min_ = stats_['min']
        mean = stats_['mean']
        std  = stats_['std']
        shape = stats_['shape']
    except KeyError as ex:
        stat_keys = stats_.keys()  # NOQA
        ut.printex(ex, key_list=['stat_keys', 'stats_', 'data'])
        raise

    #int_fmt = lambda num: util.num_fmt(int(num))
    def float_fmt(num):
        return util_num.num_fmt(float(num))

    def tup_fmt(tup):
        return str(tup)
    fmttup = (float_fmt(min_), float_fmt(max_), float_fmt(mean), float_fmt(std), tup_fmt(shape))
    lll = ' ' * len(lbl)
    if mode == 0:
        prefmtstr = r'''
        {label} stats & min ; max = %s ; %s\\
        {space}       & mean; std = %s ; %s\\
        {space}       & shape = %s \\'''
    if mode == 1:
        prefmtstr = r'''
        {label} stats & min  = $%s$\\
        {space}       & max  = $%s$\\
        {space}       & mean = $%s$\\
        {space}       & std  = $%s$\\
        {space}       & shape = $%s$\\'''
    fmtstr = prefmtstr.format(label=lbl, space=lll)
    latex_str = textwrap.dedent(fmtstr % fmttup).strip('\n') + '\n'
    return latex_str


def latex_scalar(lbl, data):
    return (r'%s & %s\\' % (lbl, util_num.num_fmt(data))) + '\n'


def make_stats_tabular():
    'tabular for dipslaying statistics'
    pass


def ensure_rowvec(arr):
    arr = np.array(arr)
    arr.shape = (1, arr.size)
    return arr


def ensure_colvec(arr):
    arr = np.array(arr)
    arr.shape = (arr.size, 1)
    return arr


def escape_latex(text):
    r"""
    Args:
        text (str): a plain text message

    Returns:
        str: the message escaped to appear correctly in LaTeX

    References:
        http://stackoverflow.com/questions/16259923/how-can-i-escape-characters
    """
    conv = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\^{}',
        '\\': r'\textbackslash{}',
        '<': r'\textless',
        '>': r'\textgreater',
    }
    import six
    regex = re.compile('|'.join(re.escape(six.text_type(key)) for key in sorted(conv.keys(), key=lambda item: - len(item))))
    return regex.sub(lambda match: conv[match.group()], text)


def replace_all(str_, repltups):
    ret = str_
    for ser, rep in repltups:
        ret = re.sub(ser, rep, ret)
    return ret


def make_score_tabular(
        row_lbls, col_lbls, values, title=None, out_of=None, bold_best=False,
        flip=False, bigger_is_better=True, multicol_lbls=None, FORCE_INT=False,
        precision=None, SHORTEN_ROW_LBLS=False, col_align='l', col_sep='|',
        multicol_sep='|', centerline=True, astable=False, table_position='',
        AUTOFIX_LATEX=True, **kwargs):
    r"""
    makes a LaTeX tabular for displaying scores or errors

    Args:
        row_lbls (list of str):
        col_lbls (list of str):
        values (ndarray):
        title (str):  (default = None)
        out_of (None): (default = None)
        bold_best (bool): (default = True)
        flip (bool): (default = False)
        table_position (str) : eg '[h]'

    Returns:
        str: tabular_str

    CommandLine:
        python -m utool.util_latex --test-make_score_tabular:0 --show
        python -m utool.util_latex --test-make_score_tabular:1 --show
        python -m utool.util_latex --test-make_score_tabular:2 --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_latex import *  # NOQA
        >>> import utool as ut
        >>> row_lbls = ['config1', 'config2']
        >>> col_lbls = [r'score \leq 1', 'metric2']
        >>> values = np.array([[1.2, 2], [3.2, 4]])
        >>> title = 'title'
        >>> out_of = 10
        >>> bold_best = True
        >>> flip = False
        >>> tabular_str = make_score_tabular(row_lbls, col_lbls, values, title, out_of, bold_best, flip)
        >>> result = tabular_str
        >>> print(result)
        >>> ut.quit_if_noshow()
        >>> render_latex_text(tabular_str)

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_latex import *  # NOQA
        >>> import utool as ut
        >>> row_lbls = ['config1']
        >>> col_lbls = [r'score \leq 1', 'metric2']
        >>> values = np.array([[1.2, 2]])
        >>> title = 'title'
        >>> out_of = 10
        >>> bold_best = True
        >>> flip = False
        >>> tabular_str = make_score_tabular(row_lbls, col_lbls, values, title, out_of, bold_best, flip)
        >>> result = tabular_str
        >>> print(result)
        >>> ut.quit_if_noshow()
        >>> render_latex_text(tabular_str)

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_latex import *  # NOQA
        >>> import utool as ut
        >>> row_lbls = ['config1', 'config2']
        >>> col_lbls = [r'score \leq 1', 'metric2', 'foobar']
        >>> multicol_lbls = [('spam', 1), ('eggs', 2)]
        >>> values = np.array([[1.2, 2, -3], [3.2, 4, -2]])
        >>> title = 'title'
        >>> out_of = 10
        >>> bold_best = True
        >>> flip = False
        >>> tabular_str = make_score_tabular(row_lbls, col_lbls, values, title, out_of, bold_best, flip, multicol_lbls=multicol_lbls)
        >>> result = tabular_str
        >>> print(result)
        >>> ut.quit_if_noshow()
        >>> render_latex_text(tabular_str)
    """
    import utool as ut
    if flip:
        bigger_is_better = not bigger_is_better
        flip_repltups = [
            ('<=', '>'),
            ('>', '<='),
            ('\\leq', '\\gt'),
            ('\\geq', '\\lt'),
            ('score', 'error')
        ]
        col_lbls = [replace_all(lbl, flip_repltups) for lbl in col_lbls]
        if title is not None:
            title = replace_all(title, flip_repltups)
        if out_of is not None:
            values = out_of - values

    # Abbreviate based on common substrings
    common_rowlbl = None
    if SHORTEN_ROW_LBLS:
        if isinstance(row_lbls, list):
            row_lbl_list = row_lbls
        else:
            row_lbl_list = row_lbls.flatten().tolist()
        # Split the rob labels into the alg components
        #algcomp_list = [lbl.split(')_') for lbl in row_lbl_list]
        longest = long_substr(row_lbl_list)
        common_strs = []
        while len(longest) > 10:
            common_strs += [longest]
            row_lbl_list = [row.replace(longest, '...') for row in row_lbl_list]
            longest = long_substr(row_lbl_list)
        common_rowlbl = ('...'.join(common_strs)).replace(')_', ')_\n')
        row_lbls = row_lbl_list
        if len(row_lbl_list) == 1:
            common_rowlbl = row_lbl_list[0]
            row_lbls = ['0']

    # Stack values into a tabular body
    # TODO: need ability to specify datatypes
    def ensurelist(row_values):
        try:
            return row_values.tolist()
        except AttributeError:
            return row_values

    if False:
        # Numpy formatting
        def padvec(shape=(1, 1)):
            pad = np.array([[' ' for c in range(shape[1])] for r in range(shape[0])])
            return pad
        col_lbls = ensure_rowvec(col_lbls)
        row_lbls = ensure_colvec(row_lbls)
        _0 = np.vstack([padvec(), row_lbls])
        _1 = np.vstack([col_lbls, values])
        body = np.hstack([_0, _1])
        body = [[str_ for str_ in row] for row in body]
    else:
        assert len(row_lbls) == len(values)
        body = [[' '] + col_lbls]
        body += [[row_lbl] + ensurelist(row_values) for row_lbl, row_values in zip(row_lbls, values)]
    #import utool as ut
    # Fix things in each body cell
    DO_PERCENT = True
    try:
        for r in range(len(body)):
            for c in range(len(body[0])):
                # In data land
                if r > 0 and c > 0:
                    if precision is not None:
                        # Hack
                        if ut.is_float(body[r][c]):
                            fmtstr = '%.' + str(precision) + 'f'
                            body[r][c] = fmtstr % (float(body[r][c]),)
                    # Force integer
                    if FORCE_INT:
                        body[r][c] = str(int(float(body[r][c])))
                body[r][c] = str(body[r][c])
                # Remove bad formatting;
                if AUTOFIX_LATEX:
                    body[r][c] = escape_latex(body[r][c])
    except Exception as ex:
        import utool as ut
        print('len(row_lbls) = %r' % (len(row_lbls),))
        print('len(col_lbls) = %r' % (len(col_lbls),))
        print('len(values) = %r' % (values,))
        print('ut.depth_profile(values) = %r' % (ut.depth_profile(values),))
        ut.printex(ex, keys=['r', 'c'])
        raise

    # Bold the best values
    if bold_best:
        best_col_scores = values.max(0) if bigger_is_better else values.min(0)
        rows_to_bold = [np.where(values[:, colx] == best_col_scores[colx])[0]
                        for colx in range(len(values.T))]
        for colx, rowx_list in enumerate(rows_to_bold):
            for rowx in rowx_list:
                body[rowx + 1][colx + 1] = '\\txtbf{' + body[rowx + 1][colx + 1] + '}'

    # More fixing after the bold is in place
    for r in range(len(body)):
        for c in range(len(body[0])):
            # In data land
            if r > 0 and c > 0:
                if out_of is not None:
                    body[r][c] = body[r][c] + '/' + str(out_of)
                    if DO_PERCENT:
                        percent = ' = %.1f%%' % float(100 * values[r - 1, c - 1] / out_of)
                        body[r][c] += escape_latex(percent)

    # Align columns for pretty printing
    body = np.array(body)
    ALIGN_BODY = True
    if ALIGN_BODY:
        new_body_cols = []
        for col in body.T:
            colstrs = list(map(str, ensurelist(col)))
            collens = list(map(len, colstrs))
            maxlen = max(collens)
            newcols = [str_ + (' ' * (maxlen - len(str_))) for str_ in colstrs]
            new_body_cols += [newcols]
        body = np.array(new_body_cols).T

    # Build Body (and row layout)
    HLINE_SEP = True
    rowvalsep = ''
    colvalsep = ' & '
    endl = '\\\\\n'
    hline = r'\hline'
    #extra_rowsep_pos_list = [1]  # rows to insert an extra hline after
    extra_rowsep_pos_list = []  # rows to insert an extra hline after
    if HLINE_SEP:
        rowvalsep = hline + '\n'
    # rowstr list holds blocks of rows
    rowstr_list = [colvalsep.join(row) + endl for row in body]
    #rowstr_list = [row[0] + rowlbl_sep + colvalsep.join(row[1:]) + endl for row in body]
    #rowstr_list = [(
    #    ('' if len(row) == 0 else row[0])
    #    if len(row) <= 1 else
    #    row[0] + rowlblcol_sep + colvalsep.join(row[1:]) + endl)
    #    for row in body]
    rowsep_list = [rowvalsep for row in rowstr_list[0:-1]]  # should be len 1 less than rowstr_list
    # Insert multicolumn names
    if multicol_lbls is not None:
        # TODO: label of the row labels
        multicol_sep
        multicols = [latex_multicolumn(multicol, size, 'c' + multicol_sep) for multicol, size in multicol_lbls]
        multicol_str = latex_multirow('', 2) + colvalsep + colvalsep.join(multicols) + endl
        ncols = sum([tup[1] for tup in multicol_lbls])
        mcol_sep = '\\cline{2-%d}\n' % (ncols + 1,)
        rowstr_list = [multicol_str] + rowstr_list
        rowsep_list = [mcol_sep] + rowsep_list
        #extra_rowsep_pos_list += [1]

    # Insert title
    if title is not None and not astable:
        tex_title = latex_multicolumn(title, len(body[0])) + endl
        rowstr_list = [tex_title] + rowstr_list
        rowsep_list = [rowvalsep] + rowsep_list
        #extra_rowsep_pos_list += [2]

    # Apply an extra hline (for label)
    #extra_rowsep_pos_list = []
    for pos in sorted(extra_rowsep_pos_list)[::-1]:
        rowstr_list.insert(pos, '')
        rowsep_list.insert(pos, rowvalsep)
    #tabular_body = rowvalsep.join(rowstr_list)
    from six.moves import zip_longest
    tabular_body = ''.join([row if sep is None else row + sep for row, sep in zip_longest(rowstr_list, rowsep_list)])

    # Build Column Layout
    col_align_list = [col_align] * len(body[0])
    #extra_collayoutsep_pos_list = [1]
    extra_collayoutsep_pos_list = []
    for pos in  sorted(extra_collayoutsep_pos_list)[::-1]:
        col_align_list.insert(pos, '')
    #col_layaout_sep_list = rowlblcol_sep  # TODO

    rowlblcol_sep = '|'
    # Build build internal seprations between column alignments
    # Defaults to just the normal col_sep
    col_align_sep_list = [col_sep] * (len(col_align_list) - 1)
    # Adjust for the separations between row labels and the actual row data
    if len(col_align_sep_list) > 0:
        col_align_sep_list[0] = rowlblcol_sep
    # Continue multicolumn sepratation
    if multicol_lbls is not None:
        multicol_offsets = ut.cumsum(ut.get_list_column(multicol_lbls, 1))
        for offset in multicol_offsets:
            if offset < len(col_align_sep_list):
                col_align_sep_list[offset] = multicol_sep

    from six.moves import zip_longest
    _tmp = [ut.filter_Nones(tup) for tup in zip_longest(col_align_list, col_align_sep_list)]
    col_layout = ''.join(ut.flatten(_tmp))

    #if len(col_align_list) > 1:
    #    col_layout = col_align_list[0] + rowlblcol_sep + col_sep.join(col_align_list[1:])
    #else:
    #    col_layout = col_sep.join(col_align_list)

    tabular_head = (r'\begin{tabular}{|%s|}' % col_layout) + '\n'
    tabular_tail = r'\end{tabular}'

    if centerline:
        tabular_head = r'\centerline{' + '\n' + tabular_head
        tabular_tail = tabular_tail + '}'

    if astable:
        #tabular_head = r'\begin{centering}' + '\n' + tabular_head
        tabular_head = r'\centering' + '\n' + tabular_head
        tabular_head = r'\begin{table}' + table_position + '\n' + tabular_head

        lblstr = latex_sanitize_command_name(kwargs.get('label', title))
        caption = title
        if AUTOFIX_LATEX:
            caption = escape_latex(caption)
        caption = '\n% ---\n' + caption + '\n% ---\n'
        #tabular_head = r'\end{centering}' + '\n' + tabular_head
        tabular_tail = tabular_tail + '\n\\caption[%s]{%s}\n\\label{tbl:%s}\n\\end{table}' % (lblstr, caption, lblstr)

    tabular_str = rowvalsep.join([tabular_head, tabular_body, tabular_tail])
    topsep = '\\hline\n' if True else '\\toprule\n'
    botsep = '\\hline\n' if True else '\\bottomrule\n'
    tabular_str = tabular_head + topsep + tabular_body + botsep + tabular_tail

    if common_rowlbl is not None:
        #tabular_str += escape_latex('\n\nThe following parameters were held fixed:\n' + common_rowlbl)
        pass
    return tabular_str


def get_latex_figure_str2(fpath_list, cmdname, **kwargs):
    """ hack for candidacy """
    import utool as ut
    from os.path import relpath
    # Make relative paths
    if kwargs.pop('relpath', True):
        start = ut.truepath('~/latex/crall-candidacy-2015')
        fpath_list = [relpath(fpath, start) for fpath in fpath_list]
    cmdname = ut.latex_sanitize_command_name(cmdname)

    kwargs['caption_str'] = kwargs.get('caption_str', cmdname)
    figure_str  = ut.get_latex_figure_str(fpath_list, **kwargs)
    latex_block = ut.latex_newcommand(cmdname, figure_str)
    return latex_block


def get_latex_figure_str(fpath_list, caption_str=None, label_str=None,
                         width_str=r'\textwidth', height_str=None, nCols=None,
                         dpath=None, colpos_sep=' ', nlsep='',
                         use_sublbls=None, use_frame=False):
    r"""
    Args:
        fpath_list (list):
        dpath (str): directory relative to main tex file

    Returns:
        str: figure_str

    CommandLine:
        python -m utool.util_latex --test-get_latex_figure_str

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_latex import *  # NOQA
        >>> fpath_list = ['figures/foo.png']
        >>> figure_str = get_latex_figure_str(fpath_list)
        >>> result = str(figure_str)
        >>> print(result)
    """
    import utool as ut

    if nCols is None:
        nCols = len(fpath_list)

    USE_SUBFIGURE = True

    if width_str is not None:
        colwidth = (1.0 / nCols)
        if USE_SUBFIGURE:
            colwidth *= .95
            graphics_sizestr = ('%.2f' % (colwidth,)) + width_str
        else:
            graphics_sizestr = '[width=%.1f%s]' % (colwidth, width_str)
    elif height_str is not None:
        graphics_sizestr = '[height=%s]' % (height_str)
    else:
        graphics_sizestr =  ''

    if dpath is not None:
        fpath_list = [ut.relpath_unix(fpath_, dpath) for fpath_ in fpath_list]

    if USE_SUBFIGURE:
        # References: https://en.wikibooks.org/wiki/LaTeX/Floats,_Figures_and_Captions#Subfloats
        # TODO ? http://tex.stackexchange.com/questions/159290/how-can-i-place-a-vertical-rule-between-subfigures
        # Use subfigures
        graphics_list = []
        sublbl_prefix = label_str if label_str is not None else ''
        for count, fpath in enumerate(fpath_list):
            """
            print(', '.join([str(x) + ':' + chr(x) for x in range(65, 123)]))
            print(', '.join([str(x) + ':' + chr(x) for x in range(97, 123)]))
            """
            CHRLBLS = True
            if CHRLBLS:
                #subchar = chr(97 + count)
                subchar = chr(65 + count)
            else:
                subchar = str(count)
            parts = []
            subfigure_str = ''
            if len(fpath_list) > 1:
                parts.append('\\begin{subfigure}[h]{' + graphics_sizestr + '}')
                parts.append('\\centering')
            graphics_part = '\\includegraphics[width=%s]{%s}' % (width_str, fpath,)
            if use_frame:
                parts.append('\\fbox{%s}' % (graphics_part,))
            else:
                parts.append(graphics_part)
            if use_sublbls is True or use_sublbls is None and len(fpath_list) > 1:
                parts.append('\\caption{}\\label{sub:' + sublbl_prefix + subchar + '}')
            if len(fpath_list) > 1:
                parts.append('\\end{subfigure}')
            subfigure_str = ''.join(parts)
            graphics_list.append(subfigure_str)
    else:
        if True:
            graphics_list = [
                r'\includegraphics%s{%s}\captionof{figure}{%s}' % (
                    graphics_sizestr, fpath, 'fd',
                    #'(' + str(count) + ')'
                    #'(' + chr(97 + count) + ')'
                )
                for count, fpath in enumerate(fpath_list)]
        else:
            graphics_list = [r'\includegraphics%s{%s}' % (graphics_sizestr, fpath,) for fpath in fpath_list]
        #graphics_list = [r'\includegraphics%s{%s}' % (graphics_sizestr, fpath,) ]
    #nRows = len(graphics_list) // nCols

    # Add separators
    NL = '\n'
    if USE_SUBFIGURE:
        col_spacer_mid = NL + '~~' + '% --' + NL
        col_spacer_end = NL + r'\\' + '% --' + NL
    else:
        col_spacer_mid = NL + '&' + NL
        col_spacer_end = NL + r'\\' + nlsep + NL
    sep_list = [
        col_spacer_mid  if count % nCols > 0 else col_spacer_end
        for count in range(1, len(graphics_list) + 1)
    ]
    if len(sep_list) > 0:
        sep_list[-1] = ''
    graphics_list_ = [graphstr + sep for graphstr, sep in zip(graphics_list, sep_list)]

    #graphics_body = '\n&\n'.join(graphics_list)
    graphics_body = ''.join(graphics_list_)
    header_str = colpos_sep.join(['c'] * nCols)

    if USE_SUBFIGURE:
        figure_body = graphics_body
    else:
        figure_body =  ut.codeblock(
            r'''
            \begin{tabular}{%s}
            %s
            \end{tabular}
            '''
        ) % (header_str, graphics_body)
    if caption_str is not None:
        #tabular_body += '\n\caption{\\footnotesize{%s}}' % (caption_str,)
        if label_str is not None:
            figure_body += '\n\\caption[%s]{%s}' % (label_str, caption_str,)
        else:
            figure_body += '\n\\caption{%s}' % (caption_str,)
    if label_str is not None:
        figure_body += '\n\\label{fig:%s}' % (label_str,)
    #figure_fmtstr = ut.codeblock(
    #    r'''
    #    \begin{figure*}
    #    \begin{center}
    #    %s
    #    \end{center}
    #    \end{figure*}
    #    '''
    #)
    figure_fmtstr = ut.codeblock(
        r'''
        \begin{figure}[ht!]
        \centering
        %s
        \end{figure}
        '''
    )
    figure_str = figure_fmtstr % (figure_body)
    return figure_str


def long_substr(strlist):
    # Longest common substring
    # http://stackoverflow.com/questions/2892931/longest-common-substring-from-more-than-two-strings-python
    substr = ''
    if len(strlist) > 1 and len(strlist[0]) > 0:
        for i in range(len(strlist[0])):
            for j in range(len(strlist[0]) - i + 1):
                if j > len(substr) and is_substr(strlist[0][i:i + j], strlist):
                    substr = strlist[0][i:i + j]
    return substr


def is_substr(find, strlist):
    if len(strlist) < 1 and len(find) < 1:
        return False
    for i in range(len(strlist)):
        if find not in strlist[i]:
            return False
    return True


def tabular_join(tabular_body_list, nCols=2):
    dedent = textwrap.dedent
    tabular_head = dedent(r'''
    \begin{tabular}{|l|l|}
    ''')
    tabular_tail = dedent(r'''
    \end{tabular}
    ''')
    hline = ''.join([r'\hline', '\n'])
    tabular_body = hline.join(tabular_body_list)
    tabular = hline.join([tabular_head, tabular_body, tabular_tail])
    return tabular


def latex_newcommand(command_name, command_text, num_args=0):
    newcmd_str = '\\newcommand{\\' + command_name + '}'
    if num_args > 0:
        newcmd_str += '[%d]' % (num_args,)
    newcmd_str += '{'
    if '\n' in command_text:
        newcmd_str += '\n'
    newcmd_str += command_text
    if '\n' in command_text:
        newcmd_str += '\n'
    newcmd_str += '}'
    return newcmd_str


def latex_sanitize_command_name(_cmdname):
    r"""
    Args:
        _cmdname (?):

    Returns:
        ?: command_name

    CommandLine:
        python -m utool.util_latex --exec-latex_sanitize_command_name

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_latex import *  # NOQA
        >>> _cmdname = '#foo bar.'
        >>> command_name = latex_sanitize_command_name(_cmdname)
        >>> result = ('command_name = %s' % (str(command_name),))
        >>> print(result)
        FooBar
    """
    import utool as ut
    command_name = _cmdname
    try:
        def subroman(match):
            import roman
            try:
                groupdict = match.groupdict()
                num = int(groupdict['num'])
                if num == 0:
                    return ''
                return roman.toRoman(num)
            except Exception as ex:
                ut.printex(ex, keys=['groupdict'])
                raise
        command_name = re.sub(ut.named_field('num', r'\d+'), subroman, command_name)
    except ImportError as ex:
        if ut.SUPER_STRICT:
            ut.printex(ex)
            raise
    # remove numbers
    command_name = re.sub(r'[\d' + re.escape('#()[]{}.') + ']', '', command_name)
    # Remove _ for camel case
    #def to_camel_case(str_list):
    #    # hacky
    #    return ''.join([str_ if len(str_) < 1 else str_[0].upper() + str_[1:] for str_ in str_list])
    #command_name = to_cammel_case(re.split('[_ ]', command_name)[::2])
    str_list = re.split('[_ ]', command_name)
    #command_name = to_cammel_case(str_list)
    command_name = ut.to_camel_case('_'.join(str_list), mixed=True)
    return command_name


if __name__ == '__main__':
    """
    CommandLine:
        python -m utool.util_latex
        python -m utool.util_latex --allexamples
        python -m utool.util_latex --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
