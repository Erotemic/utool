# -*- coding: utf-8 -*-
"""
TODO: box and whisker
http://tex.stackexchange.com/questions/115210/boxplot-in-latex
"""
from __future__ import absolute_import, division, print_function
from six.moves import range, map, zip
import os
import re
import textwrap
try:
    import numpy as np
except ImportError:
    pass
from os.path import join, splitext, dirname  # NOQA
from utool import util_cplat
from utool import util_path
from utool import util_num
from utool import util_dev
from utool import util_io
from utool.util_dbg import printex
from utool.util_inject import inject
print, print_, printDBG, rrr, profile = inject(__name__, '[latex]')

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


def make_full_document(text, title=None, preamp_decl={}):
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
    doc_preamb = ut.codeblock(r'''
    %\documentclass{article}
    \documentclass[10pt,twocolumn,letterpaper]{article}

    \usepackage[english=nohyphenation]{hyphsubst}
    \usepackage{times}
    \usepackage{amsmath}
    \usepackage{amsthm}
    \usepackage{amssymb}
    \usepackage[usenames,dvipsnames,svgnames,table]{xcolor}
    \usepackage{mathtools}
    \usepackage{algorithm}
    \usepackage{ulem}
    \normalem
    \usepackage{graphicx,adjustbox}
    \usepackage{multirow}
    \usepackage[T1]{fontenc}
    \usepackage{booktabs}
    \usepackage[margin=1.25in]{geometry}

    %\pagenumbering{gobble}

    ''')
    if title is not None:
        preamp_decl['title'] = title

    decl_lines = [r'\{key}{{{val}}}'.format(key=key, val=val) for key, val in preamp_decl.items()]
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


def render_latex_text(input_text, nest_in_doc=False, appname='utool', verbose=True):
    """ testing function """
    import utool as ut
    dpath = ut.get_app_resource_dir(appname)
    # put a latex framgent in a full document
    print(input_text)
    pdf_fpath = ut.compile_latex_text(input_text, dpath=dpath, verbose=verbose)
    ut.startfile(pdf_fpath)
    return pdf_fpath


def compile_latex_text(input_text, fnum=1, dpath=None, verbose=True, fname=None, title=None, nest_in_doc=None, **kwargs):
    r"""
    pdflatex -shell-escape --synctex=-1 -src-specials -interaction=nonstopmode /home/joncrall/code/ibeis/tmptex/latex_formatter_temp.tex

    CommandLine:
        python -m utool.util_latex --test-compile_latex_text --show

    Example1:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_latex import *  # NOQA
        >>> import utool as ut
        >>> verbose = True
        >>> #dpath = '/home/joncrall/code/ibeis/aidchallenge'
        >>> dpath = dirname(ut.grab_test_imgpath())
        >>> #ut.vd(dpath)
        >>> orig_fpath_list = ut.list_images(dpath, fullpath=True)
        >>> figure_str = ut.get_latex_figure_str(orig_fpath_list, width_str='2.4in', nCols=2)
        >>> input_text = figure_str
        >>> pdf_fpath = ut.compile_latex_text(input_text, dpath=dpath, verbose=verbose)
        >>> output_pdf_fpath = ut.compress_pdf(pdf_fpath)
        >>> print(pdf_fpath)
        >>> ut.quit_if_noshow()
        >>> ut.startfile(pdf_fpath)

        fpath_list
        def clipwhite_ondisk(fpath_in):
            import utool as ut
            import vtool as vt
            fpath_out = ut.augpath(fpath_in, '_clipwhite')
            img = vt.imread(fpath_in)
            thresh = 128
            fillval = [255, 255, 255]
            cropped_img = vt.crop_out_imgfill(img, fillval=fillval, thresh=thresh)
            vt.imwrite(fpath_out, cropped_img)
            return fpath_out
        fpath_list_ = [clipwhite_ondisk(fpath) for fpath in fpath_list]
        tmpfig = join(dpath, 'tmpfig')
        ut.ensuredir(tmpfig)
        # Weirdness
        from os.path import *
        new_fpath_list = []
        for fpath in fpath_list_:
            fname, ext = splitext(basename(fpath))
            fname_ = ut.hashstr(fname, alphabet=ut.ALPHABET_16) + ext
            fpath_ = join(tmpfig, fname_)
            ut.move(fpath, fpath_)
            new_fpath_list.append(fpath_)
        new_rel_fpath_list = [ut.relpath_unix(fpath_, dpath) for fpath_ in new_fpath_list]

    """
    #import pylab as plt
    #import matplotlib as mpl
    #verbose = True
    if nest_in_doc or (nest_in_doc is None and input_text.find('documentclass') == -1):
        text = make_full_document(input_text, title=title)
    #text = make_full_document(input_text, title=title)
    cwd = os.getcwd()
    if dpath is None:
        text_dir = join(cwd, 'tmptex')
    else:
        text_dir = dpath
    util_path.ensuredir(text_dir, verbose=verbose)
    if fname is None:
        fname = 'latex_formatter_temp'
    text_fname = fname + '.tex'
    #text_fname = 'latex_formatter_temp.tex'
    text_fpath = join(text_dir, text_fname)
    pdf_fpath = splitext(text_fpath)[0] + '.pdf'
    #jpg_fpath = splitext(text_fpath)[0] + '.jpg'
    try:
        os.chdir(text_dir)
        util_io.write_to(text_fpath, text)
        pdflatex_args = ('pdflatex', '-shell-escape', '--synctex=-1', '-src-specials', '-interaction=nonstopmode')
        args = pdflatex_args + (text_fpath,)
        util_cplat.cmd(*args, verbose=verbose, **kwargs)
        assert util_path.checkpath(pdf_fpath, verbose=verbose), 'latex failed'
    except Exception as ex:
        ut.printex(ex, 'LATEX ERROR')
    finally:
        os.chdir(cwd)
    return pdf_fpath


def render(input_text, fnum=1, dpath=None, verbose=True):
    """
    fixme or remove
    """
    import pylab as plt
    import matplotlib as mpl
    #verbose = True
    text = make_full_document(input_text)
    cwd = os.getcwd()
    if dpath is None:
        text_dir = join(cwd, 'tmptex')
    else:
        text_dir = dpath
    util_path.ensuredir(text_dir, verbose=verbose)
    text_fname = 'latex_formatter_temp.tex'
    text_fpath = join(text_dir, text_fname)
    pdf_fpath = splitext(text_fpath)[0] + '.pdf'
    jpg_fpath = splitext(text_fpath)[0] + '.jpg'
    try:
        os.chdir(text_dir)
        util_io.write_to(text_fpath, text)
        pdflatex_args = ('pdflatex', '-shell-escape', '--synctex=-1', '-src-specials', '-interaction=nonstopmode')
        args = pdflatex_args + (text_fpath,)
        util_cplat.cmd(*args, verbose=verbose)
        assert util_path.checkpath(pdf_fpath, verbose=verbose), 'latex failed'
        # convert latex pdf to jpeg
        util_cplat.cmd('convert', '-density', '300', pdf_fpath, '-quality', '90', jpg_fpath, verbose=verbose)
        assert util_path.checkpath(jpg_fpath, verbose=verbose), 'imgmagick failed'
        tex_img = plt.imread(jpg_fpath)
        # Crop img bbox
        nonwhite_x = np.where(tex_img.flatten() != 255)[0]
        nonwhite_rows = nonwhite_x // tex_img.shape[1]
        nonwhite_cols = nonwhite_x % tex_img.shape[1]
        x1 = nonwhite_cols.min()
        y1 = nonwhite_rows.min()
        x2 = nonwhite_cols.max()
        y2 = nonwhite_rows.max()
        #util.embed()
        cropped = tex_img[y1:y2, x1:x2]
        fig = plt.figure(fnum)
        fig.clf()
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(cropped, cmap=mpl.cm.gray)
        #mpl.rc('text', usetex=True)
        #mpl.rc('font', family='serif')
        #plt.figure()
        #plt.text(9, 3.4, text, size=12)
    except Exception as ex:
        print('LATEX ERROR')
        print(text)
        print(ex)
        print('LATEX ERROR')
        pass
    finally:
        os.chdir(cwd)
        #if dpath is None:
        #    if util_path.checkpath(text_dir, verbose=verbose):
        #        util_path.delete(text_dir)


def latex_multicolumn(data, ncol=2, alignstr='|c|'):
    data = escape_latex(data)
    return r'\multicolumn{%d}{%s}{%s}' % (ncol, alignstr, data)


def latex_multirow(data, nrow=2):
    return r'\multirow{%d}{*}{%s}' % (nrow, data)


def latex_get_stats(lbl, data, mode=0):
    stats_ = util_dev.get_stats(data)
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
        printex(ex, key_list=['stat_keys', 'stats_', 'data'])
        raise

    #int_fmt = lambda num: util.num_fmt(int(num))
    float_fmt = lambda num: util_num.num_fmt(float(num))
    tup_fmt = lambda tup: str(tup)
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


def escape_latex(unescaped_latex_str):
    ret = unescaped_latex_str
    ret = ret.replace('#', '\\#')
    ret = ret.replace('%', '\\%')
    ret = ret.replace('_', '\\_')
    return ret


def replace_all(str_, repltups):
    ret = str_
    for ser, rep in repltups:
        ret = re.sub(ser, rep, ret)
    return ret


def make_score_tabular(row_lbls, col_lbls, values, title=None, out_of=None,
                       bold_best=False, flip=False, bigger_is_better=True,
                       multicol_lbls=None, FORCE_INT=True, precision=None,
                       SHORTEN_ROW_LBLS=False, col_align='l', centerline=True):
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
        >>> col_lbls = ['score \leq 1', 'metric2']
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
        >>> col_lbls = ['score \leq 1', 'metric2']
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
        >>> col_lbls = ['score \leq 1', 'metric2', 'foobar']
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
    AUTOFIX_LATEX = True
    DO_PERCENT = True
    try:
        for r in range(len(body)):
            for c in range(len(body[0])):
                # In data land
                if r > 0 and c > 0:
                    if precision is not None:
                        # Hack
                        import utool as ut
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
    rowsep = ''
    colsep = ' & '
    endl = '\\\\\n'
    hline = r'\hline'
    #extra_rowsep_pos_list = [1]  # rows to insert an extra hline after
    extra_rowsep_pos_list = []  # rows to insert an extra hline after
    if HLINE_SEP:
        rowsep = hline + '\n'
    # rowstr list holds blocks of rows
    rowstr_list = [colsep.join(row) + endl for row in body]
    rowsep_list = [rowsep for row in rowstr_list[0:-1]]  # should be len 1 less than rowstr_list
    # Insert multicolumn names
    if multicol_lbls is not None:
        # TODO: label of the row labels
        multicol_str = latex_multirow('', 2) + colsep + colsep.join([latex_multicolumn(multicol, size, 'c|') for multicol, size in multicol_lbls]) + endl
        ncols = sum([tup[1] for tup in multicol_lbls])
        mcol_sep = '\\cline{2-%d}\n' % (ncols + 1,)
        rowstr_list = [multicol_str] + rowstr_list
        rowsep_list = [mcol_sep] + rowsep_list
        #extra_rowsep_pos_list += [1]

    # Insert title
    if title is not None:
        tex_title = latex_multicolumn(title, len(body[0])) + endl
        rowstr_list = [tex_title] + rowstr_list
        rowsep_list = [rowsep] + rowsep_list
        #extra_rowsep_pos_list += [2]

    # Apply an extra hline (for label)
    #extra_rowsep_pos_list = []
    for pos in sorted(extra_rowsep_pos_list)[::-1]:
        rowstr_list.insert(pos, '')
        rowsep_list.insert(pos, rowsep)
    #tabular_body = rowsep.join(rowstr_list)
    from six.moves import zip_longest
    tabular_body = ''.join([row if sep is None else row + sep for row, sep in zip_longest(rowstr_list, rowsep_list)])

    # Build Column Layout
    col_layout_sep = '|'
    col_layout_list = [col_align] * len(body[0])
    #extra_collayoutsep_pos_list = [1]
    extra_collayoutsep_pos_list = []
    for pos in  sorted(extra_collayoutsep_pos_list)[::-1]:
        col_layout_list.insert(pos, '')
    col_layout = col_layout_sep.join(col_layout_list)

    if centerline:
        tabular_head = r'\centerline{' + '\n' + (r'\begin{tabular}{|%s|}' % col_layout) + '\n'
        tabular_tail = r'\end{tabular}' + '\n' + '}'
    else:
        tabular_head = (r'\begin{tabular}{|%s|}' % col_layout) + '\n'
        tabular_tail = r'\end{tabular}'

    tabular_str = rowsep.join([tabular_head, tabular_body, tabular_tail])
    topsep = '\\hline\n' if True else '\\toprule\n'
    botsep = '\\hline\n' if True else '\\bottomrule\n'
    tabular_str = tabular_head + topsep + tabular_body + botsep + tabular_tail

    if common_rowlbl is not None:
        #tabular_str += escape_latex('\n\nThe following parameters were held fixed:\n' + common_rowlbl)
        pass
    return tabular_str


def get_latex_figure_str(fpath_list, caption_str=None, label_str=None, width_str=r'\textwdith', height_str=None, nCols=None, dpath=None):
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
        >>> # build test data
        >>> fpath_list = ['figures/foo.png']
        >>> # execute function
        >>> figure_str = get_latex_figure_str(fpath_list)
        >>> # verify results
        >>> result = str(figure_str)
        >>> print(result)
    """
    import utool as ut
    if width_str is not None:
        graphics_sizestr = '[width=%s]' % (width_str)
    elif height_str is not None:
        graphics_sizestr = '[height=%s]' % (height_str)
    else:
        graphics_sizestr =  ''

    if dpath is not None:
        fpath_list = [ut.relpath_unix(fpath_, dpath) for fpath_ in fpath_list]
    graphics_list = [r'\includegraphics%s{%s}' % (graphics_sizestr, fpath,) for fpath in fpath_list]

    if nCols is None:
        nCols = len(graphics_list)
    #nRows = len(graphics_list) // nCols

    # Add separators
    sep_list = [
        '\n&\n' if count % nCols > 0 else '\n\\\\\n'
        for count in range(1, len(graphics_list) + 1)
    ]
    sep_list[-1] = ''
    graphics_list_ = [graphstr + sep for graphstr, sep in zip(graphics_list, sep_list)]

    #graphics_body = '\n&\n'.join(graphics_list)
    graphics_body = ''.join(graphics_list_)
    header_str = ' '.join(['c'] * nCols)

    tabular_body =  ut.codeblock(
        r'''
        \begin{tabular}{%s}
        %s
        \end{tabular}
        '''
    ) % (header_str, graphics_body)
    if caption_str is not None:
        #tabular_body += '\n\caption{\\footnotesize{%s}}' % (caption_str,)
        tabular_body += '\n\caption{%s}' % (caption_str,)
    if label_str is not None:
        tabular_body += '\n\label{fig:%s}' % (label_str,)
    figure_fmtstr = ut.codeblock(
        r'''
        \begin{figure*}
        \begin{center}
        %s
        \end{center}
        \end{figure*}
        '''
    )
    figure_str = figure_fmtstr % (tabular_body)
    return figure_str


def _tabular_header_and_footer(col_layout):
    tabular_head = textwrap.dedent(r'\begin{tabular}{|%s|}' % col_layout)
    tabular_tail = textwrap.dedent(r'\end{tabular}')
    return tabular_head, tabular_tail


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


def latex_sanatize_command_name(command_name):
    import re
    import utool as ut
    # remove numbers
    command_name = re.sub(ut.regex_or(['\\d']), '', command_name)
    # Remove _ for cammel case
    def to_cammel_case(str_list):
        return ''.join([str_[0].upper() + str_[1:] for str_ in str_list])
    command_name = to_cammel_case(command_name.split('_'))
    return command_name


def get_bibtex_dict(bib_fpath):
    r"""
    Args:
        bib_fpath (str):

    Returns:
        dict: bibtex_dict

    CommandLine:
        python -m utool.util_latex --test-get_bibtex_dict

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_latex import *  # NOQA
        >>> import utool as ut
        >>> bib_fpath = ut.truepath('~/latex/crall-candidacy-2015/My_Library_clean.bib')
        >>> bibtex_dict = get_bibtex_dict(bib_fpath)
        >>> result = ('bibtex_dict = %s' % (str(bibtex_dict),))
        >>> print(result)
    """
    import bibtexparser
    import utool as ut
    bibtex_str   = ut.readfrom(bib_fpath, verbose=False)
    bib_database = bibtexparser.loads(bibtex_str)
    bibtex_dict  = bib_database.get_entry_dict()
    return bibtex_dict

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
