# Auto-harvested from matplotlib
import typing
import matplotlib

def matplotlib_ExecutableNotFoundError_add_note(self: 'any') -> 'any_computed':
    r"""
    Exception.add_note(note) --
    add a note to the exception
    Keywords: matplotlib.ExecutableNotFoundError, add_note
    """
    output_var = input_var.add_note()

def matplotlib_ExecutableNotFoundError_with_traceback(self: 'any') -> 'any_computed':
    r"""
    Exception.with_traceback(tb) --
    set self.__traceback__ to tb and return self.
    Keywords: matplotlib.ExecutableNotFoundError, with_traceback
    """
    output_var = input_var.with_traceback()

def matplotlib_MatplotlibDeprecationWarning_add_note(self: 'any') -> 'any_computed':
    r"""
    Exception.add_note(note) --
    add a note to the exception
    Keywords: matplotlib.MatplotlibDeprecationWarning, add_note
    """
    output_var = input_var.add_note()

def matplotlib_MatplotlibDeprecationWarning_with_traceback(self: 'any') -> 'any_computed':
    r"""
    Exception.with_traceback(tb) --
    set self.__traceback__ to tb and return self.
    Keywords: matplotlib.MatplotlibDeprecationWarning, with_traceback
    """
    output_var = input_var.with_traceback()

def matplotlib_RcParams_clear(self: 'any') -> 'any_computed':
    r"""
    D.clear() -> None.  Remove all items from D.
    Keywords: matplotlib.RcParams, clear
    """
    output_var = input_var.clear()

def matplotlib_RcParams_copy(self: 'any') -> 'any_computed':
    r"""
    Copy this RcParams instance.
    Keywords: matplotlib.RcParams, copy
    """
    output_var = input_var.copy()

def matplotlib_RcParams_find_all(self: 'any') -> 'any_computed':
    r"""
    Return the subset of this RcParams dictionary whose keys match,
    using :func:`re.search`, the given ``pattern``.
    .. note::
    Keywords: matplotlib.RcParams, find_all
    """
    output_var = input_var.find_all()

def matplotlib_RcParams_fromkeys(iterable: 'any') -> 'any_computed':
    r"""
    Create a new dictionary with keys from iterable and values set to value.
    Keywords: matplotlib.RcParams, fromkeys
    """
    output_var = input_var.fromkeys()

def matplotlib_RcParams_get(self: 'any') -> 'any_computed':
    r"""
    D.get(k[,d]) -> D[k] if k in D, else d.  d defaults to None.
    Keywords: matplotlib.RcParams, get
    """
    output_var = input_var.get()

def matplotlib_RcParams_items(self: 'any') -> 'any_computed':
    r"""
    D.items() -> a set-like object providing a view on D's items
    Keywords: matplotlib.RcParams, items
    """
    output_var = input_var.items()

def matplotlib_RcParams_keys(self: 'any') -> 'any_computed':
    r"""
    D.keys() -> a set-like object providing a view on D's keys
    Keywords: matplotlib.RcParams, keys
    """
    output_var = input_var.keys()

def matplotlib_RcParams_pop(self: 'any') -> 'any_computed':
    r"""
    D.pop(k[,d]) -> v, remove specified key and return the corresponding value.
    If key is not found, d is returned if given, otherwise KeyError is raised.
    Keywords: matplotlib.RcParams, pop
    """
    output_var = input_var.pop()

def matplotlib_RcParams_popitem(self: 'any') -> 'any_computed':
    r"""
    D.popitem() -> (k, v), remove and return some (key, value) pair
    as a 2-tuple; but raise KeyError if D is empty.
    Keywords: matplotlib.RcParams, popitem
    """
    output_var = input_var.popitem()

def matplotlib_RcParams_setdefault(self: 'any') -> 'any_computed':
    r"""
    D.setdefault(k[,d]) -> D.get(k,d), also set D[k]=d if k not in D
    Keywords: matplotlib.RcParams, setdefault
    """
    output_var = input_var.setdefault()

def matplotlib_RcParams_update(self: 'any') -> 'any_computed':
    r"""
    D.update([E, ]**F) -> None.  Update D from mapping/iterable E and F.
    If E present and has a .keys() method, does:     for k in E.keys(): D[k] = E[k]
    If E present and lacks .keys() method, does:     for (k, v) in E: D[k] = v
    In either case, this is followed by: for k, v in F.items(): D[k] = v
    Keywords: matplotlib.RcParams, update
    """
    output_var = input_var.update()

def matplotlib_RcParams_values(self: 'any') -> 'any_computed':
    r"""
    D.values() -> an object providing a view on D's values
    Keywords: matplotlib.RcParams, values
    """
    output_var = input_var.values()

def matplotlib_artist_Artist_add_callback(self: 'any') -> 'any_computed':
    r"""
    Add a callback function that will be called whenever one of the
    `.Artist`'s properties changes.
    Parameters
    ----------
    Keywords: matplotlib.artist.Artist, add_callback
    """
    output_var = input_var.add_callback()

def matplotlib_artist_Artist_contains(self: 'any') -> 'any_computed':
    r"""
    Test whether the artist contains the mouse event.
    Parameters
    ----------
    mouseevent : `~matplotlib.backend_bases.MouseEvent`
    Keywords: matplotlib.artist.Artist, contains
    """
    output_var = input_var.contains()

def matplotlib_artist_Artist_convert_xunits(self: 'any') -> 'any_computed':
    r"""
    Convert *x* using the unit type of the xaxis.
    If the artist is not contained in an Axes or if the xaxis does not
    have units, *x* itself is returned.
    Keywords: matplotlib.artist.Artist, convert_xunits
    """
    output_var = input_var.convert_xunits()

def matplotlib_artist_Artist_convert_yunits(self: 'any') -> 'any_computed':
    r"""
    Convert *y* using the unit type of the yaxis.
    If the artist is not contained in an Axes or if the yaxis does not
    have units, *y* itself is returned.
    Keywords: matplotlib.artist.Artist, convert_yunits
    """
    output_var = input_var.convert_yunits()

def matplotlib_artist_Artist_draw(self: 'any') -> 'any_computed':
    r"""
    Draw the Artist (and its children) using the given renderer.
    This has no effect if the artist is not visible (`.Artist.get_visible`
    returns False).
    Keywords: matplotlib.artist.Artist, draw
    """
    output_var = input_var.draw()

def matplotlib_artist_Artist_findobj(self: 'any') -> 'any_computed':
    r"""
    Find artist objects.
    Recursively find all `.Artist` instances contained in the artist.
    Parameters
    Keywords: matplotlib.artist.Artist, findobj
    """
    output_var = input_var.findobj()

def matplotlib_artist_Artist_format_cursor_data(self: 'any') -> 'any_computed':
    r"""
    Return a string representation of *data*.
    .. note::
        This method is intended to be overridden by artist subclasses.
        As an end-user of Matplotlib you will most likely not call this
    Keywords: matplotlib.artist.Artist, format_cursor_data
    """
    output_var = input_var.format_cursor_data()

def matplotlib_artist_Artist_get_agg_filter(self: 'any') -> 'any_computed':
    r"""
    Return filter function to be used for agg filter.
    Keywords: matplotlib.artist.Artist, get_agg_filter
    """
    output_var = input_var.get_agg_filter()

def matplotlib_artist_Artist_get_alpha(self: 'any') -> 'any_computed':
    r"""
    Return the alpha value used for blending - not supported on all
    backends.
    Keywords: matplotlib.artist.Artist, get_alpha
    """
    output_var = input_var.get_alpha()

def matplotlib_artist_Artist_get_animated(self: 'any') -> 'any_computed':
    r"""
    Return whether the artist is animated.
    Keywords: matplotlib.artist.Artist, get_animated
    """
    output_var = input_var.get_animated()

def matplotlib_artist_Artist_get_children(self: 'any') -> 'any_computed':
    r"""
    Return a list of the child `.Artist`\s of this `.Artist`.
    Keywords: matplotlib.artist.Artist, get_children
    """
    output_var = input_var.get_children()

def matplotlib_artist_Artist_get_clip_box(self: 'any') -> 'any_computed':
    r"""
    Return the clipbox.
    Keywords: matplotlib.artist.Artist, get_clip_box
    """
    output_var = input_var.get_clip_box()

def matplotlib_artist_Artist_get_clip_on(self: 'any') -> 'any_computed':
    r"""
    Return whether the artist uses clipping.
    Keywords: matplotlib.artist.Artist, get_clip_on
    """
    output_var = input_var.get_clip_on()

def matplotlib_artist_Artist_get_clip_path(self: 'any') -> 'any_computed':
    r"""
    Return the clip path.
    Keywords: matplotlib.artist.Artist, get_clip_path
    """
    output_var = input_var.get_clip_path()

def matplotlib_artist_Artist_get_cursor_data(self: 'any') -> 'any_computed':
    r"""
    Return the cursor data for a given event.
    .. note::
        This method is intended to be overridden by artist subclasses.
        As an end-user of Matplotlib you will most likely not call this
    Keywords: matplotlib.artist.Artist, get_cursor_data
    """
    output_var = input_var.get_cursor_data()

def matplotlib_artist_Artist_get_figure(self: 'any') -> 'any_computed':
    r"""
    Return the `.Figure` or `.SubFigure` instance the artist belongs to.
    Parameters
    ----------
    root : bool, default=False
    Keywords: matplotlib.artist.Artist, get_figure
    """
    output_var = input_var.get_figure()

def matplotlib_artist_Artist_get_gid(self: 'any') -> 'any_computed':
    r"""
    Return the group id.
    Keywords: matplotlib.artist.Artist, get_gid
    """
    output_var = input_var.get_gid()

def matplotlib_artist_Artist_get_in_layout(self: 'any') -> 'any_computed':
    r"""
    Return boolean flag, ``True`` if artist is included in layout
    calculations.
    E.g. :ref:`constrainedlayout_guide`,
    `.Figure.tight_layout()`, and
    Keywords: matplotlib.artist.Artist, get_in_layout
    """
    output_var = input_var.get_in_layout()

def matplotlib_artist_Artist_get_label(self: 'any') -> 'any_computed':
    r"""
    Return the label used for this artist in the legend.
    Keywords: matplotlib.artist.Artist, get_label
    """
    output_var = input_var.get_label()

def matplotlib_artist_Artist_get_mouseover(self: 'any') -> 'any_computed':
    r"""
    Return whether this artist is queried for custom context information
    when the mouse cursor moves over it.
    Keywords: matplotlib.artist.Artist, get_mouseover
    """
    output_var = input_var.get_mouseover()

def matplotlib_artist_Artist_get_path_effects(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.artist.Artist, get_path_effects
    """
    output_var = input_var.get_path_effects()

def matplotlib_artist_Artist_get_picker(self: 'any') -> 'any_computed':
    r"""
    Return the picking behavior of the artist.
    The possible values are described in `.Artist.set_picker`.
    See Also
    Keywords: matplotlib.artist.Artist, get_picker
    """
    output_var = input_var.get_picker()

def matplotlib_artist_Artist_get_rasterized(self: 'any') -> 'any_computed':
    r"""
    Return whether the artist is to be rasterized.
    Keywords: matplotlib.artist.Artist, get_rasterized
    """
    output_var = input_var.get_rasterized()

def matplotlib_artist_Artist_get_sketch_params(self: 'any') -> 'any_computed':
    r"""
    Return the sketch parameters for the artist.
    Returns
    -------
    tuple or None
    Keywords: matplotlib.artist.Artist, get_sketch_params
    """
    output_var = input_var.get_sketch_params()

def matplotlib_artist_Artist_get_snap(self: 'any') -> 'any_computed':
    r"""
    Return the snap setting.
    See `.set_snap` for details.
    Keywords: matplotlib.artist.Artist, get_snap
    """
    output_var = input_var.get_snap()

def matplotlib_artist_Artist_get_tightbbox(self: 'any') -> 'any_computed':
    r"""
    Like `.Artist.get_window_extent`, but includes any clipping.
    Parameters
    ----------
    renderer : `~matplotlib.backend_bases.RendererBase` subclass, optional
    Keywords: matplotlib.artist.Artist, get_tightbbox
    """
    output_var = input_var.get_tightbbox()

def matplotlib_artist_Artist_get_transform(self: 'any') -> 'any_computed':
    r"""
    Return the `.Transform` instance used by this artist.
    Keywords: matplotlib.artist.Artist, get_transform
    """
    output_var = input_var.get_transform()

def matplotlib_artist_Artist_get_transformed_clip_path_and_affine(self: 'any') -> 'any_computed':
    r"""
    Return the clip path with the non-affine part of its
    transformation applied, and the remaining affine part of its
    transformation.
    Keywords: matplotlib.artist.Artist, get_transformed_clip_path_and_affine
    """
    output_var = input_var.get_transformed_clip_path_and_affine()

def matplotlib_artist_Artist_get_url(self: 'any') -> 'any_computed':
    r"""
    Return the url.
    Keywords: matplotlib.artist.Artist, get_url
    """
    output_var = input_var.get_url()

def matplotlib_artist_Artist_get_visible(self: 'any') -> 'any_computed':
    r"""
    Return the visibility.
    Keywords: matplotlib.artist.Artist, get_visible
    """
    output_var = input_var.get_visible()

def matplotlib_artist_Artist_get_window_extent(self: 'any') -> 'any_computed':
    r"""
    Get the artist's bounding box in display space.
    The bounding box' width and height are nonnegative.
    Subclasses should override for inclusion in the bounding box
    Keywords: matplotlib.artist.Artist, get_window_extent
    """
    output_var = input_var.get_window_extent()

def matplotlib_artist_Artist_get_zorder(self: 'any') -> 'any_computed':
    r"""
    Return the artist's zorder.
    Keywords: matplotlib.artist.Artist, get_zorder
    """
    output_var = input_var.get_zorder()

def matplotlib_artist_Artist_have_units(self: 'any') -> 'any_computed':
    r"""
    Return whether units are set on any axis.
    Keywords: matplotlib.artist.Artist, have_units
    """
    output_var = input_var.have_units()

def matplotlib_artist_Artist_is_transform_set(self: 'any') -> 'any_computed':
    r"""
    Return whether the Artist has an explicitly set transform.
    This is *True* after `.set_transform` has been called.
    Keywords: matplotlib.artist.Artist, is_transform_set
    """
    output_var = input_var.is_transform_set()

def matplotlib_artist_Artist_pchanged(self: 'any') -> 'any_computed':
    r"""
    Call all of the registered callbacks.
    This function is triggered internally when a property is changed.
    See Also
    Keywords: matplotlib.artist.Artist, pchanged
    """
    output_var = input_var.pchanged()

def matplotlib_artist_Artist_pick(self: 'any') -> 'any_computed':
    r"""
    Process a pick event.
    Each child artist will fire a pick event if *mouseevent* is over
    the artist and the artist has picker set.
    Keywords: matplotlib.artist.Artist, pick
    """
    output_var = input_var.pick()

def matplotlib_artist_Artist_pickable(self: 'any') -> 'any_computed':
    r"""
    Return whether the artist is pickable.
    See Also
    --------
    .Artist.set_picker, .Artist.get_picker, .Artist.pick
    Keywords: matplotlib.artist.Artist, pickable
    """
    output_var = input_var.pickable()

def matplotlib_artist_Artist_properties(self: 'any') -> 'any_computed':
    r"""
    Return a dictionary of all the properties of the artist.
    Keywords: matplotlib.artist.Artist, properties
    """
    output_var = input_var.properties()

def matplotlib_artist_Artist_remove(self: 'any') -> 'any_computed':
    r"""
    Remove the artist from the figure if possible.
    The effect will not be visible until the figure is redrawn, e.g.,
    with `.FigureCanvasBase.draw_idle`.  Call `~.axes.Axes.relim` to
    update the Axes limits if desired.
    Keywords: matplotlib.artist.Artist, remove
    """
    output_var = input_var.remove()

def matplotlib_artist_Artist_remove_callback(self: 'any') -> 'any_computed':
    r"""
    Remove a callback based on its observer id.
    See Also
    --------
    add_callback
    Keywords: matplotlib.artist.Artist, remove_callback
    """
    output_var = input_var.remove_callback()

def matplotlib_artist_Artist_set(self: 'any') -> 'any_computed':
    r"""
    Set multiple properties at once.
    Supported properties are
    Properties:
    Keywords: matplotlib.artist.Artist, set
    """
    output_var = input_var.set()

def matplotlib_artist_Artist_set_agg_filter(self: 'any') -> 'any_computed':
    r"""
    Set the agg filter.
    Parameters
    ----------
    filter_func : callable
    Keywords: matplotlib.artist.Artist, set_agg_filter
    """
    output_var = input_var.set_agg_filter()

def matplotlib_artist_Artist_set_alpha(self: 'any') -> 'any_computed':
    r"""
    Set the alpha value used for blending - not supported on all backends.
    Parameters
    ----------
    alpha : float or None
    Keywords: matplotlib.artist.Artist, set_alpha
    """
    output_var = input_var.set_alpha()

def matplotlib_artist_Artist_set_animated(self: 'any') -> 'any_computed':
    r"""
    Set whether the artist is intended to be used in an animation.
    If True, the artist is excluded from regular drawing of the figure.
    You have to call `.Figure.draw_artist` / `.Axes.draw_artist`
    explicitly on the artist. This approach is used to speed up animations
    Keywords: matplotlib.artist.Artist, set_animated
    """
    output_var = input_var.set_animated()

def matplotlib_artist_Artist_set_clip_box(self: 'any') -> 'any_computed':
    r"""
    Set the artist's clip `.Bbox`.
    Parameters
    ----------
    clipbox : `~matplotlib.transforms.BboxBase` or None
    Keywords: matplotlib.artist.Artist, set_clip_box
    """
    output_var = input_var.set_clip_box()

def matplotlib_artist_Artist_set_clip_on(self: 'any') -> 'any_computed':
    r"""
    Set whether the artist uses clipping.
    When False, artists will be visible outside the Axes which
    can lead to unexpected results.
    Keywords: matplotlib.artist.Artist, set_clip_on
    """
    output_var = input_var.set_clip_on()

def matplotlib_artist_Artist_set_clip_path(self: 'any') -> 'any_computed':
    r"""
    Set the artist's clip path.
    Parameters
    ----------
    path : `~matplotlib.patches.Patch` or `.Path` or `.TransformedPath` or None
    Keywords: matplotlib.artist.Artist, set_clip_path
    """
    output_var = input_var.set_clip_path()

def matplotlib_artist_Artist_set_figure(self: 'any') -> 'any_computed':
    r"""
    Set the `.Figure` or `.SubFigure` instance the artist belongs to.
    Parameters
    ----------
    fig : `~matplotlib.figure.Figure` or `~matplotlib.figure.SubFigure`
    Keywords: matplotlib.artist.Artist, set_figure
    """
    output_var = input_var.set_figure()

def matplotlib_artist_Artist_set_gid(self: 'any') -> 'any_computed':
    r"""
    Set the (group) id for the artist.
    Parameters
    ----------
    gid : str
    Keywords: matplotlib.artist.Artist, set_gid
    """
    output_var = input_var.set_gid()

def matplotlib_artist_Artist_set_in_layout(self: 'any') -> 'any_computed':
    r"""
    Set if artist is to be included in layout calculations,
    E.g. :ref:`constrainedlayout_guide`,
    `.Figure.tight_layout()`, and
    ``fig.savefig(fname, bbox_inches='tight')``.
    Keywords: matplotlib.artist.Artist, set_in_layout
    """
    output_var = input_var.set_in_layout()

def matplotlib_artist_Artist_set_label(self: 'any') -> 'any_computed':
    r"""
    Set a label that will be displayed in the legend.
    Parameters
    ----------
    s : object
    Keywords: matplotlib.artist.Artist, set_label
    """
    output_var = input_var.set_label()

def matplotlib_artist_Artist_set_mouseover(self: 'any') -> 'any_computed':
    r"""
    Set whether this artist is queried for custom context information when
    the mouse cursor moves over it.
    Parameters
    ----------
    Keywords: matplotlib.artist.Artist, set_mouseover
    """
    output_var = input_var.set_mouseover()

def matplotlib_artist_Artist_set_path_effects(self: 'any') -> 'any_computed':
    r"""
    Set the path effects.
    Parameters
    ----------
    path_effects : list of `.AbstractPathEffect`
    Keywords: matplotlib.artist.Artist, set_path_effects
    """
    output_var = input_var.set_path_effects()

def matplotlib_artist_Artist_set_picker(self: 'any') -> 'any_computed':
    r"""
    Define the picking behavior of the artist.
    Parameters
    ----------
    picker : None or bool or float or callable
    Keywords: matplotlib.artist.Artist, set_picker
    """
    output_var = input_var.set_picker()

def matplotlib_artist_Artist_set_rasterized(self: 'any') -> 'any_computed':
    r"""
    Force rasterized (bitmap) drawing for vector graphics output.
    Rasterized drawing is not supported by all artists. If you try to
    enable this on an artist that does not support it, the command has no
    effect and a warning will be issued.
    Keywords: matplotlib.artist.Artist, set_rasterized
    """
    output_var = input_var.set_rasterized()

def matplotlib_artist_Artist_set_sketch_params(self: 'any') -> 'any_computed':
    r"""
    Set the sketch parameters.
    Parameters
    ----------
    scale : float, optional
    Keywords: matplotlib.artist.Artist, set_sketch_params
    """
    output_var = input_var.set_sketch_params()

def matplotlib_artist_Artist_set_snap(self: 'any') -> 'any_computed':
    r"""
    Set the snapping behavior.
    Snapping aligns positions with the pixel grid, which results in
    clearer images. For example, if a black line of 1px width was
    defined at a position in between two pixels, the resulting image
    Keywords: matplotlib.artist.Artist, set_snap
    """
    output_var = input_var.set_snap()

def matplotlib_artist_Artist_set_transform(self: 'any') -> 'any_computed':
    r"""
    Set the artist transform.
    Parameters
    ----------
    t : `~matplotlib.transforms.Transform`
    Keywords: matplotlib.artist.Artist, set_transform
    """
    output_var = input_var.set_transform()

def matplotlib_artist_Artist_set_url(self: 'any') -> 'any_computed':
    r"""
    Set the url for the artist.
    Parameters
    ----------
    url : str
    Keywords: matplotlib.artist.Artist, set_url
    """
    output_var = input_var.set_url()

def matplotlib_artist_Artist_set_visible(self: 'any') -> 'any_computed':
    r"""
    Set the artist's visibility.
    Parameters
    ----------
    b : bool
    Keywords: matplotlib.artist.Artist, set_visible
    """
    output_var = input_var.set_visible()

def matplotlib_artist_Artist_set_zorder(self: 'any') -> 'any_computed':
    r"""
    Set the zorder for the artist.  Artists with lower zorder
    values are drawn first.
    Parameters
    ----------
    Keywords: matplotlib.artist.Artist, set_zorder
    """
    output_var = input_var.set_zorder()

def matplotlib_artist_Artist_update(self: 'any') -> 'any_computed':
    r"""
    Update this artist's properties from the dict *props*.
    Parameters
    ----------
    props : dict
    Keywords: matplotlib.artist.Artist, update
    """
    output_var = input_var.update()

def matplotlib_artist_Artist_update_from(self: 'any') -> 'any_computed':
    r"""
    Copy properties from *other* to *self*.
    Keywords: matplotlib.artist.Artist, update_from
    """
    output_var = input_var.update_from()

def matplotlib_artist_ArtistInspector_aliased_name(self: 'any') -> 'any_computed':
    r"""
    Return 'PROPNAME or alias' if *s* has an alias, else return 'PROPNAME'.
    For example, for the line markerfacecolor property, which has an
    alias, return 'markerfacecolor or mfc' and for the transform
    property, which does not, return 'transform'.
    Keywords: matplotlib.artist.ArtistInspector, aliased_name
    """
    output_var = input_var.aliased_name()

def matplotlib_artist_ArtistInspector_aliased_name_rest(self: 'any') -> 'any_computed':
    r"""
    Return 'PROPNAME or alias' if *s* has an alias, else return 'PROPNAME',
    formatted for reST.
    For example, for the line markerfacecolor property, which has an
    alias, return 'markerfacecolor or mfc' and for the transform
    Keywords: matplotlib.artist.ArtistInspector, aliased_name_rest
    """
    output_var = input_var.aliased_name_rest()

def matplotlib_artist_ArtistInspector_get_aliases(self: 'any') -> 'any_computed':
    r"""
    Get a dict mapping property fullnames to sets of aliases for each alias
    in the :class:`~matplotlib.artist.ArtistInspector`.
    e.g., for lines::
    Keywords: matplotlib.artist.ArtistInspector, get_aliases
    """
    output_var = input_var.get_aliases()

def matplotlib_artist_ArtistInspector_get_setters(self: 'any') -> 'any_computed':
    r"""
    Get the attribute strings with setters for object.
    For example, for a line, return ``['markerfacecolor', 'linewidth',
    ....]``.
    Keywords: matplotlib.artist.ArtistInspector, get_setters
    """
    output_var = input_var.get_setters()

def matplotlib_artist_ArtistInspector_get_valid_values(self: 'any') -> 'any_computed':
    r"""
    Get the legal arguments for the setter associated with *attr*.
    This is done by querying the docstring of the setter for a line that
    begins with "ACCEPTS:" or ".. ACCEPTS:", and then by looking for a
    numpydoc-style documentation for the setter's first argument.
    Keywords: matplotlib.artist.ArtistInspector, get_valid_values
    """
    output_var = input_var.get_valid_values()

def matplotlib_artist_ArtistInspector_is_alias(method: 'any') -> 'any_computed':
    r"""
    Return whether the object *method* is an alias for another method.
    Keywords: matplotlib.artist.ArtistInspector, is_alias
    """
    output_var = input_var.is_alias()

def matplotlib_artist_ArtistInspector_number_of_parameters(func: 'any') -> 'any_computed':
    r"""
    Return number of parameters of the callable *func*.
    Keywords: matplotlib.artist.ArtistInspector, number_of_parameters
    """
    output_var = input_var.number_of_parameters()

def matplotlib_artist_ArtistInspector_pprint_getters(self: 'any') -> 'any_computed':
    r"""
    Return the getters and actual values as list of strings.
    Keywords: matplotlib.artist.ArtistInspector, pprint_getters
    """
    output_var = input_var.pprint_getters()

def matplotlib_artist_ArtistInspector_pprint_setters(self: 'any') -> 'any_computed':
    r"""
    If *prop* is *None*, return a list of strings of all settable
    properties and their valid values.
    If *prop* is not *None*, it is a valid property name and that
    property will be returned as a string of property : valid
    Keywords: matplotlib.artist.ArtistInspector, pprint_setters
    """
    output_var = input_var.pprint_setters()

def matplotlib_artist_ArtistInspector_pprint_setters_rest(self: 'any') -> 'any_computed':
    r"""
    If *prop* is *None*, return a list of reST-formatted strings of all
    settable properties and their valid values.
    If *prop* is not *None*, it is a valid property name and that
    property will be returned as a string of "property : valid"
    Keywords: matplotlib.artist.ArtistInspector, pprint_setters_rest
    """
    output_var = input_var.pprint_setters_rest()

def matplotlib_artist_ArtistInspector_properties(self: 'any') -> 'any_computed':
    r"""
    Return a dictionary mapping property name -> value.
    Keywords: matplotlib.artist.ArtistInspector, properties
    """
    output_var = input_var.properties()

def matplotlib_artist_Bbox_anchored(self: 'any') -> 'any_computed':
    r"""
    Return a copy of the `Bbox` anchored to *c* within *container*.
    Parameters
    ----------
    c : (float, float) or {'C', 'SW', 'S', 'SE', 'E', 'NE', ...}
    Keywords: matplotlib.artist.Bbox, anchored
    """
    output_var = input_var.anchored()

def matplotlib_artist_Bbox_contains(self: 'any') -> 'any_computed':
    r"""
    Return whether ``(x, y)`` is in the bounding box or on its edge.
    Keywords: matplotlib.artist.Bbox, contains
    """
    output_var = input_var.contains()

def matplotlib_artist_Bbox_containsx(self: 'any') -> 'any_computed':
    r"""
    Return whether *x* is in the closed (:attr:`x0`, :attr:`x1`) interval.
    Keywords: matplotlib.artist.Bbox, containsx
    """
    output_var = input_var.containsx()

def matplotlib_artist_Bbox_containsy(self: 'any') -> 'any_computed':
    r"""
    Return whether *y* is in the closed (:attr:`y0`, :attr:`y1`) interval.
    Keywords: matplotlib.artist.Bbox, containsy
    """
    output_var = input_var.containsy()

def matplotlib_artist_Bbox_corners(self: 'any') -> 'any_computed':
    r"""
    Return the corners of this rectangle as an array of points.
    Specifically, this returns the array
    ``[[x0, y0], [x0, y1], [x1, y0], [x1, y1]]``.
    Keywords: matplotlib.artist.Bbox, corners
    """
    output_var = input_var.corners()

def matplotlib_artist_Bbox_count_contains(self: 'any') -> 'any_computed':
    r"""
    Count the number of vertices contained in the `Bbox`.
    Any vertices with a non-finite x or y value are ignored.
    Parameters
    ----------
    Keywords: matplotlib.artist.Bbox, count_contains
    """
    output_var = input_var.count_contains()

def matplotlib_artist_Bbox_count_overlaps(self: 'any') -> 'any_computed':
    r"""
    Count the number of bounding boxes that overlap this one.
    Parameters
    ----------
    bboxes : sequence of `.BboxBase`
    Keywords: matplotlib.artist.Bbox, count_overlaps
    """
    output_var = input_var.count_overlaps()

def matplotlib_artist_Bbox_expanded(self: 'any') -> 'any_computed':
    r"""
    Construct a `Bbox` by expanding this one around its center by the
    factors *sw* and *sh*.
    Keywords: matplotlib.artist.Bbox, expanded
    """
    output_var = input_var.expanded()

def matplotlib_artist_Bbox_from_bounds(x0: 'any') -> 'any_computed':
    r"""
    Create a new `Bbox` from *x0*, *y0*, *width* and *height*.
    *width* and *height* may be negative.
    Keywords: matplotlib.artist.Bbox, from_bounds
    """
    output_var = input_var.from_bounds()

def matplotlib_artist_Bbox_from_extents(args: 'any') -> 'any_computed':
    r"""
    Create a new Bbox from *left*, *bottom*, *right* and *top*.
    The *y*-axis increases upwards.
    Parameters
    Keywords: matplotlib.artist.Bbox, from_extents
    """
    output_var = input_var.from_extents()

def matplotlib_artist_Bbox_frozen(self: 'any') -> 'any_computed':
    r"""
    The base class for anything that participates in the transform tree
    and needs to invalidate its parents or be invalidated.  This includes
    classes that are not really transforms, such as bounding boxes, since some
    transforms depend on bounding boxes to compute their values.
    Keywords: matplotlib.artist.Bbox, frozen
    """
    output_var = input_var.frozen()

def matplotlib_artist_Bbox_fully_contains(self: 'any') -> 'any_computed':
    r"""
    Return whether ``x, y`` is in the bounding box, but not on its edge.
    Keywords: matplotlib.artist.Bbox, fully_contains
    """
    output_var = input_var.fully_contains()

def matplotlib_artist_Bbox_fully_containsx(self: 'any') -> 'any_computed':
    r"""
    Return whether *x* is in the open (:attr:`x0`, :attr:`x1`) interval.
    Keywords: matplotlib.artist.Bbox, fully_containsx
    """
    output_var = input_var.fully_containsx()

def matplotlib_artist_Bbox_fully_containsy(self: 'any') -> 'any_computed':
    r"""
    Return whether *y* is in the open (:attr:`y0`, :attr:`y1`) interval.
    Keywords: matplotlib.artist.Bbox, fully_containsy
    """
    output_var = input_var.fully_containsy()

def matplotlib_artist_Bbox_fully_overlaps(self: 'any') -> 'any_computed':
    r"""
    Return whether this bounding box overlaps with the other bounding box,
    not including the edges.
    Parameters
    ----------
    Keywords: matplotlib.artist.Bbox, fully_overlaps
    """
    output_var = input_var.fully_overlaps()

def matplotlib_artist_Bbox_get_points(self: 'any') -> 'any_computed':
    r"""
    Get the points of the bounding box as an array of the form
    ``[[x0, y0], [x1, y1]]``.
    Keywords: matplotlib.artist.Bbox, get_points
    """
    output_var = input_var.get_points()

def matplotlib_artist_Bbox_ignore(self: 'any') -> 'any_computed':
    r"""
    Set whether the existing bounds of the box should be ignored
    by subsequent calls to :meth:`update_from_data_xy`.
    value : bool
        - When ``True``, subsequent calls to `update_from_data_xy` will
    Keywords: matplotlib.artist.Bbox, ignore
    """
    output_var = input_var.ignore()

def matplotlib_artist_Bbox_intersection(bbox1: 'any') -> 'any_computed':
    r"""
    Return the intersection of *bbox1* and *bbox2* if they intersect, or
    None if they don't.
    Keywords: matplotlib.artist.Bbox, intersection
    """
    output_var = input_var.intersection()

def matplotlib_artist_Bbox_invalidate(self: 'any') -> 'any_computed':
    r"""
    Invalidate this `TransformNode` and triggers an invalidation of its
    ancestors.  Should be called any time the transform changes.
    Keywords: matplotlib.artist.Bbox, invalidate
    """
    output_var = input_var.invalidate()

def matplotlib_artist_Bbox_mutated(self: 'any') -> 'any_computed':
    r"""
    Return whether the bbox has changed since init.
    Keywords: matplotlib.artist.Bbox, mutated
    """
    output_var = input_var.mutated()

def matplotlib_artist_Bbox_mutatedx(self: 'any') -> 'any_computed':
    r"""
    Return whether the x-limits have changed since init.
    Keywords: matplotlib.artist.Bbox, mutatedx
    """
    output_var = input_var.mutatedx()

def matplotlib_artist_Bbox_mutatedy(self: 'any') -> 'any_computed':
    r"""
    Return whether the y-limits have changed since init.
    Keywords: matplotlib.artist.Bbox, mutatedy
    """
    output_var = input_var.mutatedy()

def matplotlib_artist_Bbox_overlaps(self: 'any') -> 'any_computed':
    r"""
    Return whether this bounding box overlaps with the other bounding box.
    Parameters
    ----------
    other : `.BboxBase`
    Keywords: matplotlib.artist.Bbox, overlaps
    """
    output_var = input_var.overlaps()

def matplotlib_artist_Bbox_padded(self: 'any') -> 'any_computed':
    r"""
    Construct a `Bbox` by padding this one on all four sides.
    Parameters
    ----------
    w_pad : float
    Keywords: matplotlib.artist.Bbox, padded
    """
    output_var = input_var.padded()

def matplotlib_artist_Bbox_rotated(self: 'any') -> 'any_computed':
    r"""
    Return the axes-aligned bounding box that bounds the result of rotating
    this `Bbox` by an angle of *radians*.
    Keywords: matplotlib.artist.Bbox, rotated
    """
    output_var = input_var.rotated()

def matplotlib_artist_Bbox_set(self: 'any') -> 'any_computed':
    r"""
    Set this bounding box from the "frozen" bounds of another `Bbox`.
    Keywords: matplotlib.artist.Bbox, set
    """
    output_var = input_var.set()

def matplotlib_artist_Bbox_set_children(self: 'any') -> 'any_computed':
    r"""
    Set the children of the transform, to let the invalidation
    system know which transforms can invalidate this transform.
    Should be called from the constructor of any transforms that
    depend on other transforms.
    Keywords: matplotlib.artist.Bbox, set_children
    """
    output_var = input_var.set_children()

def matplotlib_artist_Bbox_set_points(self: 'any') -> 'any_computed':
    r"""
    Set the points of the bounding box directly from an array of the form
    ``[[x0, y0], [x1, y1]]``.  No error checking is performed, as this
    method is mainly for internal use.
    Keywords: matplotlib.artist.Bbox, set_points
    """
    output_var = input_var.set_points()

def matplotlib_artist_Bbox_shrunk(self: 'any') -> 'any_computed':
    r"""
    Return a copy of the `Bbox`, shrunk by the factor *mx*
    in the *x* direction and the factor *my* in the *y* direction.
    The lower left corner of the box remains unchanged.  Normally
    *mx* and *my* will be less than 1, but this is not enforced.
    Keywords: matplotlib.artist.Bbox, shrunk
    """
    output_var = input_var.shrunk()

def matplotlib_artist_Bbox_shrunk_to_aspect(self: 'any') -> 'any_computed':
    r"""
    Return a copy of the `Bbox`, shrunk so that it is as
    large as it can be while having the desired aspect ratio,
    *box_aspect*.  If the box coordinates are relative (i.e.
    fractions of a larger box such as a figure) then the
    physical aspect ratio of that figure is specified with
    Keywords: matplotlib.artist.Bbox, shrunk_to_aspect
    """
    output_var = input_var.shrunk_to_aspect()

def matplotlib_artist_Bbox_splitx(self: 'any') -> 'any_computed':
    r"""
    Return a list of new `Bbox` objects formed by splitting the original
    one with vertical lines at fractional positions given by *args*.
    Keywords: matplotlib.artist.Bbox, splitx
    """
    output_var = input_var.splitx()

def matplotlib_artist_Bbox_splity(self: 'any') -> 'any_computed':
    r"""
    Return a list of new `Bbox` objects formed by splitting the original
    one with horizontal lines at fractional positions given by *args*.
    Keywords: matplotlib.artist.Bbox, splity
    """
    output_var = input_var.splity()

def matplotlib_artist_Bbox_transformed(self: 'any') -> 'any_computed':
    r"""
    Construct a `Bbox` by statically transforming this one by *transform*.
    Keywords: matplotlib.artist.Bbox, transformed
    """
    output_var = input_var.transformed()

def matplotlib_artist_Bbox_translated(self: 'any') -> 'any_computed':
    r"""
    Construct a `Bbox` by translating this one by *tx* and *ty*.
    Keywords: matplotlib.artist.Bbox, translated
    """
    output_var = input_var.translated()

def matplotlib_artist_Bbox_union(bboxes: 'any') -> 'any_computed':
    r"""
    Return a `Bbox` that contains all of the given *bboxes*.
    Keywords: matplotlib.artist.Bbox, union
    """
    output_var = input_var.union()

def matplotlib_artist_Bbox_update_from_data_x(self: 'any') -> 'any_computed':
    r"""
    Update the x-bounds of the `Bbox` based on the passed in data. After
    updating, the bounds will have positive *width*, and *x0* will be the
    minimal value.
    Parameters
    Keywords: matplotlib.artist.Bbox, update_from_data_x
    """
    output_var = input_var.update_from_data_x()

def matplotlib_artist_Bbox_update_from_data_xy(self: 'any') -> 'any_computed':
    r"""
    Update the `Bbox` bounds based on the passed in *xy* coordinates.
    After updating, the bounds will have positive *width* and *height*;
    *x0* and *y0* will be the minimal values.
    Keywords: matplotlib.artist.Bbox, update_from_data_xy
    """
    output_var = input_var.update_from_data_xy()

def matplotlib_artist_Bbox_update_from_data_y(self: 'any') -> 'any_computed':
    r"""
    Update the y-bounds of the `Bbox` based on the passed in data. After
    updating, the bounds will have positive *height*, and *y0* will be the
    minimal value.
    Parameters
    Keywords: matplotlib.artist.Bbox, update_from_data_y
    """
    output_var = input_var.update_from_data_y()

def matplotlib_artist_Bbox_update_from_path(self: 'any') -> 'any_computed':
    r"""
    Update the bounds of the `Bbox` to contain the vertices of the
    provided path. After updating, the bounds will have positive *width*
    and *height*; *x0* and *y0* will be the minimal values.
    Parameters
    Keywords: matplotlib.artist.Bbox, update_from_path
    """
    output_var = input_var.update_from_path()

def matplotlib_artist_BboxBase_anchored(self: 'any') -> 'any_computed':
    r"""
    Return a copy of the `Bbox` anchored to *c* within *container*.
    Parameters
    ----------
    c : (float, float) or {'C', 'SW', 'S', 'SE', 'E', 'NE', ...}
    Keywords: matplotlib.artist.BboxBase, anchored
    """
    output_var = input_var.anchored()

def matplotlib_artist_BboxBase_contains(self: 'any') -> 'any_computed':
    r"""
    Return whether ``(x, y)`` is in the bounding box or on its edge.
    Keywords: matplotlib.artist.BboxBase, contains
    """
    output_var = input_var.contains()

def matplotlib_artist_BboxBase_containsx(self: 'any') -> 'any_computed':
    r"""
    Return whether *x* is in the closed (:attr:`x0`, :attr:`x1`) interval.
    Keywords: matplotlib.artist.BboxBase, containsx
    """
    output_var = input_var.containsx()

def matplotlib_artist_BboxBase_containsy(self: 'any') -> 'any_computed':
    r"""
    Return whether *y* is in the closed (:attr:`y0`, :attr:`y1`) interval.
    Keywords: matplotlib.artist.BboxBase, containsy
    """
    output_var = input_var.containsy()

def matplotlib_artist_BboxBase_corners(self: 'any') -> 'any_computed':
    r"""
    Return the corners of this rectangle as an array of points.
    Specifically, this returns the array
    ``[[x0, y0], [x0, y1], [x1, y0], [x1, y1]]``.
    Keywords: matplotlib.artist.BboxBase, corners
    """
    output_var = input_var.corners()

def matplotlib_artist_BboxBase_count_contains(self: 'any') -> 'any_computed':
    r"""
    Count the number of vertices contained in the `Bbox`.
    Any vertices with a non-finite x or y value are ignored.
    Parameters
    ----------
    Keywords: matplotlib.artist.BboxBase, count_contains
    """
    output_var = input_var.count_contains()

def matplotlib_artist_BboxBase_count_overlaps(self: 'any') -> 'any_computed':
    r"""
    Count the number of bounding boxes that overlap this one.
    Parameters
    ----------
    bboxes : sequence of `.BboxBase`
    Keywords: matplotlib.artist.BboxBase, count_overlaps
    """
    output_var = input_var.count_overlaps()

def matplotlib_artist_BboxBase_expanded(self: 'any') -> 'any_computed':
    r"""
    Construct a `Bbox` by expanding this one around its center by the
    factors *sw* and *sh*.
    Keywords: matplotlib.artist.BboxBase, expanded
    """
    output_var = input_var.expanded()

def matplotlib_artist_BboxBase_frozen(self: 'any') -> 'any_computed':
    r"""
    The base class for anything that participates in the transform tree
    and needs to invalidate its parents or be invalidated.  This includes
    classes that are not really transforms, such as bounding boxes, since some
    transforms depend on bounding boxes to compute their values.
    Keywords: matplotlib.artist.BboxBase, frozen
    """
    output_var = input_var.frozen()

def matplotlib_artist_BboxBase_fully_contains(self: 'any') -> 'any_computed':
    r"""
    Return whether ``x, y`` is in the bounding box, but not on its edge.
    Keywords: matplotlib.artist.BboxBase, fully_contains
    """
    output_var = input_var.fully_contains()

def matplotlib_artist_BboxBase_fully_containsx(self: 'any') -> 'any_computed':
    r"""
    Return whether *x* is in the open (:attr:`x0`, :attr:`x1`) interval.
    Keywords: matplotlib.artist.BboxBase, fully_containsx
    """
    output_var = input_var.fully_containsx()

def matplotlib_artist_BboxBase_fully_containsy(self: 'any') -> 'any_computed':
    r"""
    Return whether *y* is in the open (:attr:`y0`, :attr:`y1`) interval.
    Keywords: matplotlib.artist.BboxBase, fully_containsy
    """
    output_var = input_var.fully_containsy()

def matplotlib_artist_BboxBase_fully_overlaps(self: 'any') -> 'any_computed':
    r"""
    Return whether this bounding box overlaps with the other bounding box,
    not including the edges.
    Parameters
    ----------
    Keywords: matplotlib.artist.BboxBase, fully_overlaps
    """
    output_var = input_var.fully_overlaps()

def matplotlib_artist_BboxBase_get_points(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.artist.BboxBase, get_points
    """
    output_var = input_var.get_points()

def matplotlib_artist_BboxBase_intersection(bbox1: 'any') -> 'any_computed':
    r"""
    Return the intersection of *bbox1* and *bbox2* if they intersect, or
    None if they don't.
    Keywords: matplotlib.artist.BboxBase, intersection
    """
    output_var = input_var.intersection()

def matplotlib_artist_BboxBase_invalidate(self: 'any') -> 'any_computed':
    r"""
    Invalidate this `TransformNode` and triggers an invalidation of its
    ancestors.  Should be called any time the transform changes.
    Keywords: matplotlib.artist.BboxBase, invalidate
    """
    output_var = input_var.invalidate()

def matplotlib_artist_BboxBase_overlaps(self: 'any') -> 'any_computed':
    r"""
    Return whether this bounding box overlaps with the other bounding box.
    Parameters
    ----------
    other : `.BboxBase`
    Keywords: matplotlib.artist.BboxBase, overlaps
    """
    output_var = input_var.overlaps()

def matplotlib_artist_BboxBase_padded(self: 'any') -> 'any_computed':
    r"""
    Construct a `Bbox` by padding this one on all four sides.
    Parameters
    ----------
    w_pad : float
    Keywords: matplotlib.artist.BboxBase, padded
    """
    output_var = input_var.padded()

def matplotlib_artist_BboxBase_rotated(self: 'any') -> 'any_computed':
    r"""
    Return the axes-aligned bounding box that bounds the result of rotating
    this `Bbox` by an angle of *radians*.
    Keywords: matplotlib.artist.BboxBase, rotated
    """
    output_var = input_var.rotated()

def matplotlib_artist_BboxBase_set_children(self: 'any') -> 'any_computed':
    r"""
    Set the children of the transform, to let the invalidation
    system know which transforms can invalidate this transform.
    Should be called from the constructor of any transforms that
    depend on other transforms.
    Keywords: matplotlib.artist.BboxBase, set_children
    """
    output_var = input_var.set_children()

def matplotlib_artist_BboxBase_shrunk(self: 'any') -> 'any_computed':
    r"""
    Return a copy of the `Bbox`, shrunk by the factor *mx*
    in the *x* direction and the factor *my* in the *y* direction.
    The lower left corner of the box remains unchanged.  Normally
    *mx* and *my* will be less than 1, but this is not enforced.
    Keywords: matplotlib.artist.BboxBase, shrunk
    """
    output_var = input_var.shrunk()

def matplotlib_artist_BboxBase_shrunk_to_aspect(self: 'any') -> 'any_computed':
    r"""
    Return a copy of the `Bbox`, shrunk so that it is as
    large as it can be while having the desired aspect ratio,
    *box_aspect*.  If the box coordinates are relative (i.e.
    fractions of a larger box such as a figure) then the
    physical aspect ratio of that figure is specified with
    Keywords: matplotlib.artist.BboxBase, shrunk_to_aspect
    """
    output_var = input_var.shrunk_to_aspect()

def matplotlib_artist_BboxBase_splitx(self: 'any') -> 'any_computed':
    r"""
    Return a list of new `Bbox` objects formed by splitting the original
    one with vertical lines at fractional positions given by *args*.
    Keywords: matplotlib.artist.BboxBase, splitx
    """
    output_var = input_var.splitx()

def matplotlib_artist_BboxBase_splity(self: 'any') -> 'any_computed':
    r"""
    Return a list of new `Bbox` objects formed by splitting the original
    one with horizontal lines at fractional positions given by *args*.
    Keywords: matplotlib.artist.BboxBase, splity
    """
    output_var = input_var.splity()

def matplotlib_artist_BboxBase_transformed(self: 'any') -> 'any_computed':
    r"""
    Construct a `Bbox` by statically transforming this one by *transform*.
    Keywords: matplotlib.artist.BboxBase, transformed
    """
    output_var = input_var.transformed()

def matplotlib_artist_BboxBase_translated(self: 'any') -> 'any_computed':
    r"""
    Construct a `Bbox` by translating this one by *tx* and *ty*.
    Keywords: matplotlib.artist.BboxBase, translated
    """
    output_var = input_var.translated()

def matplotlib_artist_BboxBase_union(bboxes: 'any') -> 'any_computed':
    r"""
    Return a `Bbox` that contains all of the given *bboxes*.
    Keywords: matplotlib.artist.BboxBase, union
    """
    output_var = input_var.union()

def matplotlib_artist_IdentityTransform_contains_branch(self: 'any') -> 'any_computed':
    r"""
    Return whether the given transform is a sub-tree of this transform.
    This routine uses transform equality to identify sub-trees, therefore
    in many situations it is object id which will be used.
    Keywords: matplotlib.artist.IdentityTransform, contains_branch
    """
    output_var = input_var.contains_branch()

def matplotlib_artist_IdentityTransform_contains_branch_seperately(self: 'any') -> 'any_computed':
    r"""
    Return whether the given branch is a sub-tree of this transform on
    each separate dimension.
    A common use for this method is to identify if a transform is a blended
    transform containing an Axes' data transform. e.g.::
    Keywords: matplotlib.artist.IdentityTransform, contains_branch_seperately
    """
    output_var = input_var.contains_branch_seperately()

def matplotlib_artist_IdentityTransform_frozen(self: 'any') -> 'any_computed':
    r"""
    Return a frozen copy of this transform node.  The frozen copy will not
    be updated when its children change.  Useful for storing a previously
    known state of a transform where ``copy.deepcopy()`` might normally be
    used.
    Keywords: matplotlib.artist.IdentityTransform, frozen
    """
    output_var = input_var.frozen()

def matplotlib_artist_IdentityTransform_get_affine(self: 'any') -> 'any_computed':
    r"""
    Get the affine part of this transform.
    Keywords: matplotlib.artist.IdentityTransform, get_affine
    """
    output_var = input_var.get_affine()

def matplotlib_artist_IdentityTransform_get_matrix(self: 'any') -> 'any_computed':
    r"""
    Get the matrix for the affine part of this transform.
    Keywords: matplotlib.artist.IdentityTransform, get_matrix
    """
    output_var = input_var.get_matrix()

def matplotlib_artist_IdentityTransform_invalidate(self: 'any') -> 'any_computed':
    r"""
    Invalidate this `TransformNode` and triggers an invalidation of its
    ancestors.  Should be called any time the transform changes.
    Keywords: matplotlib.artist.IdentityTransform, invalidate
    """
    output_var = input_var.invalidate()

def matplotlib_artist_IdentityTransform_inverted(self: 'any') -> 'any_computed':
    r"""
    Return the corresponding inverse transformation.
    It holds ``x == self.inverted().transform(self.transform(x))``.
    The return value of this method should be treated as
    Keywords: matplotlib.artist.IdentityTransform, inverted
    """
    output_var = input_var.inverted()

def matplotlib_artist_IdentityTransform_set_children(self: 'any') -> 'any_computed':
    r"""
    Set the children of the transform, to let the invalidation
    system know which transforms can invalidate this transform.
    Should be called from the constructor of any transforms that
    depend on other transforms.
    Keywords: matplotlib.artist.IdentityTransform, set_children
    """
    output_var = input_var.set_children()

def matplotlib_artist_IdentityTransform_to_values(self: 'any') -> 'any_computed':
    r"""
    Return the values of the matrix as an ``(a, b, c, d, e, f)`` tuple.
    Keywords: matplotlib.artist.IdentityTransform, to_values
    """
    output_var = input_var.to_values()

def matplotlib_artist_IdentityTransform_transform(self: 'any') -> 'any_computed':
    r"""
    Apply this transformation on the given array of *values*.
    Parameters
    ----------
    values : array-like
    Keywords: matplotlib.artist.IdentityTransform, transform
    """
    output_var = input_var.transform()

def matplotlib_artist_IdentityTransform_transform_affine(self: 'any') -> 'any_computed':
    r"""
    Apply only the affine part of this transformation on the
    given array of values.
    ``transform(values)`` is always equivalent to
    ``transform_affine(transform_non_affine(values))``.
    Keywords: matplotlib.artist.IdentityTransform, transform_affine
    """
    output_var = input_var.transform_affine()

def matplotlib_artist_IdentityTransform_transform_angles(self: 'any') -> 'any_computed':
    r"""
    Transform a set of angles anchored at specific locations.
    Parameters
    ----------
    angles : (N,) array-like
    Keywords: matplotlib.artist.IdentityTransform, transform_angles
    """
    output_var = input_var.transform_angles()

def matplotlib_artist_IdentityTransform_transform_bbox(self: 'any') -> 'any_computed':
    r"""
    Transform the given bounding box.
    For smarter transforms including caching (a common requirement in
    Matplotlib), see `TransformedBbox`.
    Keywords: matplotlib.artist.IdentityTransform, transform_bbox
    """
    output_var = input_var.transform_bbox()

def matplotlib_artist_IdentityTransform_transform_non_affine(self: 'any') -> 'any_computed':
    r"""
    Apply only the non-affine part of this transformation.
    ``transform(values)`` is always equivalent to
    ``transform_affine(transform_non_affine(values))``.
    Keywords: matplotlib.artist.IdentityTransform, transform_non_affine
    """
    output_var = input_var.transform_non_affine()

def matplotlib_artist_IdentityTransform_transform_path(self: 'any') -> 'any_computed':
    r"""
    Apply the transform to `.Path` *path*, returning a new `.Path`.
    In some cases, this transform may insert curves into the path
    that began as line segments.
    Keywords: matplotlib.artist.IdentityTransform, transform_path
    """
    output_var = input_var.transform_path()

def matplotlib_artist_IdentityTransform_transform_path_affine(self: 'any') -> 'any_computed':
    r"""
    Apply the affine part of this transform to `.Path` *path*, returning a
    new `.Path`.
    ``transform_path(path)`` is equivalent to
    ``transform_path_affine(transform_path_non_affine(values))``.
    Keywords: matplotlib.artist.IdentityTransform, transform_path_affine
    """
    output_var = input_var.transform_path_affine()

def matplotlib_artist_IdentityTransform_transform_path_non_affine(self: 'any') -> 'any_computed':
    r"""
    Apply the non-affine part of this transform to `.Path` *path*,
    returning a new `.Path`.
    ``transform_path(path)`` is equivalent to
    ``transform_path_affine(transform_path_non_affine(values))``.
    Keywords: matplotlib.artist.IdentityTransform, transform_path_non_affine
    """
    output_var = input_var.transform_path_non_affine()

def matplotlib_artist_IdentityTransform_transform_point(self: 'any') -> 'any_computed':
    r"""
    Return a transformed point.
    This function is only kept for backcompatibility; the more general
    `.transform` method is capable of transforming both a list of points
    and a single point.
    Keywords: matplotlib.artist.IdentityTransform, transform_point
    """
    output_var = input_var.transform_point()

def matplotlib_artist_Path_arc(theta1: 'any') -> 'any_computed':
    r"""
    Return a `Path` for the unit circle arc from angles *theta1* to
    *theta2* (in degrees).
    *theta2* is unwrapped to produce the shortest arc within 360 degrees.
    That is, if *theta2* > *theta1* + 360, the arc will be from *theta1* to
    Keywords: matplotlib.artist.Path, arc
    """
    output_var = input_var.arc()

def matplotlib_artist_Path_circle(center: 'any') -> 'any_computed':
    r"""
    Return a `Path` representing a circle of a given radius and center.
    Parameters
    ----------
    center : (float, float), default: (0, 0)
    Keywords: matplotlib.artist.Path, circle
    """
    output_var = input_var.circle()

def matplotlib_artist_Path_cleaned(self: 'any') -> 'any_computed':
    r"""
    Return a new `Path` with vertices and codes cleaned according to the
    parameters.
    See Also
    --------
    Keywords: matplotlib.artist.Path, cleaned
    """
    output_var = input_var.cleaned()

def matplotlib_artist_Path_clip_to_bbox(self: 'any') -> 'any_computed':
    r"""
    Clip the path to the given bounding box.
    The path must be made up of one or more closed polygons.  This
    algorithm will not behave correctly for unclosed paths.
    Keywords: matplotlib.artist.Path, clip_to_bbox
    """
    output_var = input_var.clip_to_bbox()

def matplotlib_artist_Path_contains_path(self: 'any') -> 'any_computed':
    r"""
    Return whether this (closed) path completely contains the given path.
    If *transform* is not ``None``, the path will be transformed before
    checking for containment.
    Keywords: matplotlib.artist.Path, contains_path
    """
    output_var = input_var.contains_path()

def matplotlib_artist_Path_contains_point(self: 'any') -> 'any_computed':
    r"""
    Return whether the area enclosed by the path contains the given point.
    The path is always treated as closed; i.e. if the last code is not
    `CLOSEPOLY` an implicit segment connecting the last vertex to the first
    vertex is assumed.
    Keywords: matplotlib.artist.Path, contains_point
    """
    output_var = input_var.contains_point()

def matplotlib_artist_Path_contains_points(self: 'any') -> 'any_computed':
    r"""
    Return whether the area enclosed by the path contains the given points.
    The path is always treated as closed; i.e. if the last code is not
    `CLOSEPOLY` an implicit segment connecting the last vertex to the first
    vertex is assumed.
    Keywords: matplotlib.artist.Path, contains_points
    """
    output_var = input_var.contains_points()

def matplotlib_artist_Path_copy(self: 'any') -> 'any_computed':
    r"""
    Return a shallow copy of the `Path`, which will share the
    vertices and codes with the source `Path`.
    Keywords: matplotlib.artist.Path, copy
    """
    output_var = input_var.copy()

def matplotlib_artist_Path_deepcopy(self: 'any') -> 'any_computed':
    r"""
    Return a deep copy of the `Path`.  The `Path` will not be readonly,
    even if the source `Path` is.
    Parameters
    ----------
    Keywords: matplotlib.artist.Path, deepcopy
    """
    output_var = input_var.deepcopy()

def matplotlib_artist_Path_get_extents(self: 'any') -> 'any_computed':
    r"""
    Get Bbox of the path.
    Parameters
    ----------
    transform : `~matplotlib.transforms.Transform`, optional
    Keywords: matplotlib.artist.Path, get_extents
    """
    output_var = input_var.get_extents()

def matplotlib_artist_Path_hatch(hatchpattern: 'any') -> 'any_computed':
    r"""
    Given a hatch specifier, *hatchpattern*, generates a `Path` that
    can be used in a repeated hatching pattern.  *density* is the
    number of lines per unit square.
    Keywords: matplotlib.artist.Path, hatch
    """
    output_var = input_var.hatch()

def matplotlib_artist_Path_interpolated(self: 'any') -> 'any_computed':
    r"""
    Return a new path with each segment divided into *steps* parts.
    Codes other than `LINETO`, `MOVETO`, and `CLOSEPOLY` are not handled correctly.
    Parameters
    Keywords: matplotlib.artist.Path, interpolated
    """
    output_var = input_var.interpolated()

def matplotlib_artist_Path_intersects_bbox(self: 'any') -> 'any_computed':
    r"""
    Return whether this path intersects a given `~.transforms.Bbox`.
    If *filled* is True, then this also returns True if the path completely
    encloses the `.Bbox` (i.e., the path is treated as filled).
    Keywords: matplotlib.artist.Path, intersects_bbox
    """
    output_var = input_var.intersects_bbox()

def matplotlib_artist_Path_intersects_path(self: 'any') -> 'any_computed':
    r"""
    Return whether if this path intersects another given path.
    If *filled* is True, then this also returns True if one path completely
    encloses the other (i.e., the paths are treated as filled).
    Keywords: matplotlib.artist.Path, intersects_path
    """
    output_var = input_var.intersects_path()

def matplotlib_artist_Path_iter_bezier(self: 'any') -> 'any_computed':
    r"""
    Iterate over each Bézier curve (lines included) in a `Path`.
    Parameters
    ----------
    **kwargs
    Keywords: matplotlib.artist.Path, iter_bezier
    """
    output_var = input_var.iter_bezier()

def matplotlib_artist_Path_iter_segments(self: 'any') -> 'any_computed':
    r"""
    Iterate over all curve segments in the path.
    Each iteration returns a pair ``(vertices, code)``, where ``vertices``
    is a sequence of 1-3 coordinate pairs, and ``code`` is a `Path` code.
    Keywords: matplotlib.artist.Path, iter_segments
    """
    output_var = input_var.iter_segments()

def matplotlib_artist_Path_make_compound_path(args: 'any') -> 'any_computed':
    r"""
    Concatenate a list of `Path`\s into a single `Path`, removing all `STOP`\s.
    Keywords: matplotlib.artist.Path, make_compound_path
    """
    output_var = input_var.make_compound_path()

def matplotlib_artist_Path_make_compound_path_from_polys(XY: 'any') -> 'any_computed':
    r"""
    Make a compound `Path` object to draw a number of polygons with equal
    numbers of sides.
    .. plot:: gallery/misc/histogram_path.py
    Keywords: matplotlib.artist.Path, make_compound_path_from_polys
    """
    output_var = input_var.make_compound_path_from_polys()

def matplotlib_artist_Path_to_polygons(self: 'any') -> 'any_computed':
    r"""
    Convert this path to a list of polygons or polylines.  Each
    polygon/polyline is an (N, 2) array of vertices.  In other words,
    each polygon has no `MOVETO` instructions or curves.  This
    is useful for displaying in backends that do not support
    compound paths or Bézier curves.
    Keywords: matplotlib.artist.Path, to_polygons
    """
    output_var = input_var.to_polygons()

def matplotlib_artist_Path_transformed(self: 'any') -> 'any_computed':
    r"""
    Return a transformed copy of the path.
    See Also
    --------
    matplotlib.transforms.TransformedPath
    Keywords: matplotlib.artist.Path, transformed
    """
    output_var = input_var.transformed()

def matplotlib_artist_Path_unit_regular_asterisk(numVertices: 'any') -> 'any_computed':
    r"""
    Return a :class:`Path` for a unit regular asterisk with the given
    numVertices and radius of 1.0, centered at (0, 0).
    Keywords: matplotlib.artist.Path, unit_regular_asterisk
    """
    output_var = input_var.unit_regular_asterisk()

def matplotlib_artist_Path_unit_regular_polygon(numVertices: 'any') -> 'any_computed':
    r"""
    Return a :class:`Path` instance for a unit regular polygon with the
    given *numVertices* such that the circumscribing circle has radius 1.0,
    centered at (0, 0).
    Keywords: matplotlib.artist.Path, unit_regular_polygon
    """
    output_var = input_var.unit_regular_polygon()

def matplotlib_artist_Path_unit_regular_star(numVertices: 'any') -> 'any_computed':
    r"""
    Return a :class:`Path` for a unit regular star with the given
    numVertices and radius of 1.0, centered at (0, 0).
    Keywords: matplotlib.artist.Path, unit_regular_star
    """
    output_var = input_var.unit_regular_star()

def matplotlib_artist_Path_wedge(theta1: 'any') -> 'any_computed':
    r"""
    Return a `Path` for the unit circle wedge from angles *theta1* to
    *theta2* (in degrees).
    *theta2* is unwrapped to produce the shortest wedge within 360 degrees.
    That is, if *theta2* > *theta1* + 360, the wedge will be from *theta1*
    Keywords: matplotlib.artist.Path, wedge
    """
    output_var = input_var.wedge()

def matplotlib_artist_Transform_contains_branch(self: 'any') -> 'any_computed':
    r"""
    Return whether the given transform is a sub-tree of this transform.
    This routine uses transform equality to identify sub-trees, therefore
    in many situations it is object id which will be used.
    Keywords: matplotlib.artist.Transform, contains_branch
    """
    output_var = input_var.contains_branch()

def matplotlib_artist_Transform_contains_branch_seperately(self: 'any') -> 'any_computed':
    r"""
    Return whether the given branch is a sub-tree of this transform on
    each separate dimension.
    A common use for this method is to identify if a transform is a blended
    transform containing an Axes' data transform. e.g.::
    Keywords: matplotlib.artist.Transform, contains_branch_seperately
    """
    output_var = input_var.contains_branch_seperately()

def matplotlib_artist_Transform_frozen(self: 'any') -> 'any_computed':
    r"""
    Return a frozen copy of this transform node.  The frozen copy will not
    be updated when its children change.  Useful for storing a previously
    known state of a transform where ``copy.deepcopy()`` might normally be
    used.
    Keywords: matplotlib.artist.Transform, frozen
    """
    output_var = input_var.frozen()

def matplotlib_artist_Transform_get_affine(self: 'any') -> 'any_computed':
    r"""
    Get the affine part of this transform.
    Keywords: matplotlib.artist.Transform, get_affine
    """
    output_var = input_var.get_affine()

def matplotlib_artist_Transform_get_matrix(self: 'any') -> 'any_computed':
    r"""
    Get the matrix for the affine part of this transform.
    Keywords: matplotlib.artist.Transform, get_matrix
    """
    output_var = input_var.get_matrix()

def matplotlib_artist_Transform_invalidate(self: 'any') -> 'any_computed':
    r"""
    Invalidate this `TransformNode` and triggers an invalidation of its
    ancestors.  Should be called any time the transform changes.
    Keywords: matplotlib.artist.Transform, invalidate
    """
    output_var = input_var.invalidate()

def matplotlib_artist_Transform_inverted(self: 'any') -> 'any_computed':
    r"""
    Return the corresponding inverse transformation.
    It holds ``x == self.inverted().transform(self.transform(x))``.
    The return value of this method should be treated as
    Keywords: matplotlib.artist.Transform, inverted
    """
    output_var = input_var.inverted()

def matplotlib_artist_Transform_set_children(self: 'any') -> 'any_computed':
    r"""
    Set the children of the transform, to let the invalidation
    system know which transforms can invalidate this transform.
    Should be called from the constructor of any transforms that
    depend on other transforms.
    Keywords: matplotlib.artist.Transform, set_children
    """
    output_var = input_var.set_children()

def matplotlib_artist_Transform_transform(self: 'any') -> 'any_computed':
    r"""
    Apply this transformation on the given array of *values*.
    Parameters
    ----------
    values : array-like
    Keywords: matplotlib.artist.Transform, transform
    """
    output_var = input_var.transform()

def matplotlib_artist_Transform_transform_affine(self: 'any') -> 'any_computed':
    r"""
    Apply only the affine part of this transformation on the
    given array of values.
    ``transform(values)`` is always equivalent to
    ``transform_affine(transform_non_affine(values))``.
    Keywords: matplotlib.artist.Transform, transform_affine
    """
    output_var = input_var.transform_affine()

def matplotlib_artist_Transform_transform_angles(self: 'any') -> 'any_computed':
    r"""
    Transform a set of angles anchored at specific locations.
    Parameters
    ----------
    angles : (N,) array-like
    Keywords: matplotlib.artist.Transform, transform_angles
    """
    output_var = input_var.transform_angles()

def matplotlib_artist_Transform_transform_bbox(self: 'any') -> 'any_computed':
    r"""
    Transform the given bounding box.
    For smarter transforms including caching (a common requirement in
    Matplotlib), see `TransformedBbox`.
    Keywords: matplotlib.artist.Transform, transform_bbox
    """
    output_var = input_var.transform_bbox()

def matplotlib_artist_Transform_transform_non_affine(self: 'any') -> 'any_computed':
    r"""
    Apply only the non-affine part of this transformation.
    ``transform(values)`` is always equivalent to
    ``transform_affine(transform_non_affine(values))``.
    Keywords: matplotlib.artist.Transform, transform_non_affine
    """
    output_var = input_var.transform_non_affine()

def matplotlib_artist_Transform_transform_path(self: 'any') -> 'any_computed':
    r"""
    Apply the transform to `.Path` *path*, returning a new `.Path`.
    In some cases, this transform may insert curves into the path
    that began as line segments.
    Keywords: matplotlib.artist.Transform, transform_path
    """
    output_var = input_var.transform_path()

def matplotlib_artist_Transform_transform_path_affine(self: 'any') -> 'any_computed':
    r"""
    Apply the affine part of this transform to `.Path` *path*, returning a
    new `.Path`.
    ``transform_path(path)`` is equivalent to
    ``transform_path_affine(transform_path_non_affine(values))``.
    Keywords: matplotlib.artist.Transform, transform_path_affine
    """
    output_var = input_var.transform_path_affine()

def matplotlib_artist_Transform_transform_path_non_affine(self: 'any') -> 'any_computed':
    r"""
    Apply the non-affine part of this transform to `.Path` *path*,
    returning a new `.Path`.
    ``transform_path(path)`` is equivalent to
    ``transform_path_affine(transform_path_non_affine(values))``.
    Keywords: matplotlib.artist.Transform, transform_path_non_affine
    """
    output_var = input_var.transform_path_non_affine()

def matplotlib_artist_Transform_transform_point(self: 'any') -> 'any_computed':
    r"""
    Return a transformed point.
    This function is only kept for backcompatibility; the more general
    `.transform` method is capable of transforming both a list of points
    and a single point.
    Keywords: matplotlib.artist.Transform, transform_point
    """
    output_var = input_var.transform_point()

def matplotlib_artist_TransformedBbox_anchored(self: 'any') -> 'any_computed':
    r"""
    Return a copy of the `Bbox` anchored to *c* within *container*.
    Parameters
    ----------
    c : (float, float) or {'C', 'SW', 'S', 'SE', 'E', 'NE', ...}
    Keywords: matplotlib.artist.TransformedBbox, anchored
    """
    output_var = input_var.anchored()

def matplotlib_artist_TransformedBbox_contains(self: 'any') -> 'any_computed':
    r"""
    Return whether ``(x, y)`` is in the bounding box or on its edge.
    Keywords: matplotlib.artist.TransformedBbox, contains
    """
    output_var = input_var.contains()

def matplotlib_artist_TransformedBbox_containsx(self: 'any') -> 'any_computed':
    r"""
    Return whether *x* is in the closed (:attr:`x0`, :attr:`x1`) interval.
    Keywords: matplotlib.artist.TransformedBbox, containsx
    """
    output_var = input_var.containsx()

def matplotlib_artist_TransformedBbox_containsy(self: 'any') -> 'any_computed':
    r"""
    Return whether *y* is in the closed (:attr:`y0`, :attr:`y1`) interval.
    Keywords: matplotlib.artist.TransformedBbox, containsy
    """
    output_var = input_var.containsy()

def matplotlib_artist_TransformedBbox_corners(self: 'any') -> 'any_computed':
    r"""
    Return the corners of this rectangle as an array of points.
    Specifically, this returns the array
    ``[[x0, y0], [x0, y1], [x1, y0], [x1, y1]]``.
    Keywords: matplotlib.artist.TransformedBbox, corners
    """
    output_var = input_var.corners()

def matplotlib_artist_TransformedBbox_count_contains(self: 'any') -> 'any_computed':
    r"""
    Count the number of vertices contained in the `Bbox`.
    Any vertices with a non-finite x or y value are ignored.
    Parameters
    ----------
    Keywords: matplotlib.artist.TransformedBbox, count_contains
    """
    output_var = input_var.count_contains()

def matplotlib_artist_TransformedBbox_count_overlaps(self: 'any') -> 'any_computed':
    r"""
    Count the number of bounding boxes that overlap this one.
    Parameters
    ----------
    bboxes : sequence of `.BboxBase`
    Keywords: matplotlib.artist.TransformedBbox, count_overlaps
    """
    output_var = input_var.count_overlaps()

def matplotlib_artist_TransformedBbox_expanded(self: 'any') -> 'any_computed':
    r"""
    Construct a `Bbox` by expanding this one around its center by the
    factors *sw* and *sh*.
    Keywords: matplotlib.artist.TransformedBbox, expanded
    """
    output_var = input_var.expanded()

def matplotlib_artist_TransformedBbox_frozen(self: 'any') -> 'any_computed':
    r"""
    The base class for anything that participates in the transform tree
    and needs to invalidate its parents or be invalidated.  This includes
    classes that are not really transforms, such as bounding boxes, since some
    transforms depend on bounding boxes to compute their values.
    Keywords: matplotlib.artist.TransformedBbox, frozen
    """
    output_var = input_var.frozen()

def matplotlib_artist_TransformedBbox_fully_contains(self: 'any') -> 'any_computed':
    r"""
    Return whether ``x, y`` is in the bounding box, but not on its edge.
    Keywords: matplotlib.artist.TransformedBbox, fully_contains
    """
    output_var = input_var.fully_contains()

def matplotlib_artist_TransformedBbox_fully_containsx(self: 'any') -> 'any_computed':
    r"""
    Return whether *x* is in the open (:attr:`x0`, :attr:`x1`) interval.
    Keywords: matplotlib.artist.TransformedBbox, fully_containsx
    """
    output_var = input_var.fully_containsx()

def matplotlib_artist_TransformedBbox_fully_containsy(self: 'any') -> 'any_computed':
    r"""
    Return whether *y* is in the open (:attr:`y0`, :attr:`y1`) interval.
    Keywords: matplotlib.artist.TransformedBbox, fully_containsy
    """
    output_var = input_var.fully_containsy()

def matplotlib_artist_TransformedBbox_fully_overlaps(self: 'any') -> 'any_computed':
    r"""
    Return whether this bounding box overlaps with the other bounding box,
    not including the edges.
    Parameters
    ----------
    Keywords: matplotlib.artist.TransformedBbox, fully_overlaps
    """
    output_var = input_var.fully_overlaps()

def matplotlib_artist_TransformedBbox_get_points(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.artist.TransformedBbox, get_points
    """
    output_var = input_var.get_points()

def matplotlib_artist_TransformedBbox_intersection(bbox1: 'any') -> 'any_computed':
    r"""
    Return the intersection of *bbox1* and *bbox2* if they intersect, or
    None if they don't.
    Keywords: matplotlib.artist.TransformedBbox, intersection
    """
    output_var = input_var.intersection()

def matplotlib_artist_TransformedBbox_invalidate(self: 'any') -> 'any_computed':
    r"""
    Invalidate this `TransformNode` and triggers an invalidation of its
    ancestors.  Should be called any time the transform changes.
    Keywords: matplotlib.artist.TransformedBbox, invalidate
    """
    output_var = input_var.invalidate()

def matplotlib_artist_TransformedBbox_overlaps(self: 'any') -> 'any_computed':
    r"""
    Return whether this bounding box overlaps with the other bounding box.
    Parameters
    ----------
    other : `.BboxBase`
    Keywords: matplotlib.artist.TransformedBbox, overlaps
    """
    output_var = input_var.overlaps()

def matplotlib_artist_TransformedBbox_padded(self: 'any') -> 'any_computed':
    r"""
    Construct a `Bbox` by padding this one on all four sides.
    Parameters
    ----------
    w_pad : float
    Keywords: matplotlib.artist.TransformedBbox, padded
    """
    output_var = input_var.padded()

def matplotlib_artist_TransformedBbox_rotated(self: 'any') -> 'any_computed':
    r"""
    Return the axes-aligned bounding box that bounds the result of rotating
    this `Bbox` by an angle of *radians*.
    Keywords: matplotlib.artist.TransformedBbox, rotated
    """
    output_var = input_var.rotated()

def matplotlib_artist_TransformedBbox_set_children(self: 'any') -> 'any_computed':
    r"""
    Set the children of the transform, to let the invalidation
    system know which transforms can invalidate this transform.
    Should be called from the constructor of any transforms that
    depend on other transforms.
    Keywords: matplotlib.artist.TransformedBbox, set_children
    """
    output_var = input_var.set_children()

def matplotlib_artist_TransformedBbox_shrunk(self: 'any') -> 'any_computed':
    r"""
    Return a copy of the `Bbox`, shrunk by the factor *mx*
    in the *x* direction and the factor *my* in the *y* direction.
    The lower left corner of the box remains unchanged.  Normally
    *mx* and *my* will be less than 1, but this is not enforced.
    Keywords: matplotlib.artist.TransformedBbox, shrunk
    """
    output_var = input_var.shrunk()

def matplotlib_artist_TransformedBbox_shrunk_to_aspect(self: 'any') -> 'any_computed':
    r"""
    Return a copy of the `Bbox`, shrunk so that it is as
    large as it can be while having the desired aspect ratio,
    *box_aspect*.  If the box coordinates are relative (i.e.
    fractions of a larger box such as a figure) then the
    physical aspect ratio of that figure is specified with
    Keywords: matplotlib.artist.TransformedBbox, shrunk_to_aspect
    """
    output_var = input_var.shrunk_to_aspect()

def matplotlib_artist_TransformedBbox_splitx(self: 'any') -> 'any_computed':
    r"""
    Return a list of new `Bbox` objects formed by splitting the original
    one with vertical lines at fractional positions given by *args*.
    Keywords: matplotlib.artist.TransformedBbox, splitx
    """
    output_var = input_var.splitx()

def matplotlib_artist_TransformedBbox_splity(self: 'any') -> 'any_computed':
    r"""
    Return a list of new `Bbox` objects formed by splitting the original
    one with horizontal lines at fractional positions given by *args*.
    Keywords: matplotlib.artist.TransformedBbox, splity
    """
    output_var = input_var.splity()

def matplotlib_artist_TransformedBbox_transformed(self: 'any') -> 'any_computed':
    r"""
    Construct a `Bbox` by statically transforming this one by *transform*.
    Keywords: matplotlib.artist.TransformedBbox, transformed
    """
    output_var = input_var.transformed()

def matplotlib_artist_TransformedBbox_translated(self: 'any') -> 'any_computed':
    r"""
    Construct a `Bbox` by translating this one by *tx* and *ty*.
    Keywords: matplotlib.artist.TransformedBbox, translated
    """
    output_var = input_var.translated()

def matplotlib_artist_TransformedBbox_union(bboxes: 'any') -> 'any_computed':
    r"""
    Return a `Bbox` that contains all of the given *bboxes*.
    Keywords: matplotlib.artist.TransformedBbox, union
    """
    output_var = input_var.union()

def matplotlib_artist_TransformedPatchPath_frozen(self: 'any') -> 'any_computed':
    r"""
    Return a frozen copy of this transform node.  The frozen copy will not
    be updated when its children change.  Useful for storing a previously
    known state of a transform where ``copy.deepcopy()`` might normally be
    used.
    Keywords: matplotlib.artist.TransformedPatchPath, frozen
    """
    output_var = input_var.frozen()

def matplotlib_artist_TransformedPatchPath_get_affine(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.artist.TransformedPatchPath, get_affine
    """
    output_var = input_var.get_affine()

def matplotlib_artist_TransformedPatchPath_get_fully_transformed_path(self: 'any') -> 'any_computed':
    r"""
    Return a fully-transformed copy of the child path.
    Keywords: matplotlib.artist.TransformedPatchPath, get_fully_transformed_path
    """
    output_var = input_var.get_fully_transformed_path()

def matplotlib_artist_TransformedPatchPath_get_transformed_path_and_affine(self: 'any') -> 'any_computed':
    r"""
    Return a copy of the child path, with the non-affine part of
    the transform already applied, along with the affine part of
    the path necessary to complete the transformation.
    Keywords: matplotlib.artist.TransformedPatchPath, get_transformed_path_and_affine
    """
    output_var = input_var.get_transformed_path_and_affine()

def matplotlib_artist_TransformedPatchPath_get_transformed_points_and_affine(self: 'any') -> 'any_computed':
    r"""
    Return a copy of the child path, with the non-affine part of
    the transform already applied, along with the affine part of
    the path necessary to complete the transformation.  Unlike
    :meth:`get_transformed_path_and_affine`, no interpolation will
    be performed.
    Keywords: matplotlib.artist.TransformedPatchPath, get_transformed_points_and_affine
    """
    output_var = input_var.get_transformed_points_and_affine()

def matplotlib_artist_TransformedPatchPath_invalidate(self: 'any') -> 'any_computed':
    r"""
    Invalidate this `TransformNode` and triggers an invalidation of its
    ancestors.  Should be called any time the transform changes.
    Keywords: matplotlib.artist.TransformedPatchPath, invalidate
    """
    output_var = input_var.invalidate()

def matplotlib_artist_TransformedPatchPath_set_children(self: 'any') -> 'any_computed':
    r"""
    Set the children of the transform, to let the invalidation
    system know which transforms can invalidate this transform.
    Should be called from the constructor of any transforms that
    depend on other transforms.
    Keywords: matplotlib.artist.TransformedPatchPath, set_children
    """
    output_var = input_var.set_children()

def matplotlib_artist_TransformedPath_frozen(self: 'any') -> 'any_computed':
    r"""
    Return a frozen copy of this transform node.  The frozen copy will not
    be updated when its children change.  Useful for storing a previously
    known state of a transform where ``copy.deepcopy()`` might normally be
    used.
    Keywords: matplotlib.artist.TransformedPath, frozen
    """
    output_var = input_var.frozen()

def matplotlib_artist_TransformedPath_get_affine(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.artist.TransformedPath, get_affine
    """
    output_var = input_var.get_affine()

def matplotlib_artist_TransformedPath_get_fully_transformed_path(self: 'any') -> 'any_computed':
    r"""
    Return a fully-transformed copy of the child path.
    Keywords: matplotlib.artist.TransformedPath, get_fully_transformed_path
    """
    output_var = input_var.get_fully_transformed_path()

def matplotlib_artist_TransformedPath_get_transformed_path_and_affine(self: 'any') -> 'any_computed':
    r"""
    Return a copy of the child path, with the non-affine part of
    the transform already applied, along with the affine part of
    the path necessary to complete the transformation.
    Keywords: matplotlib.artist.TransformedPath, get_transformed_path_and_affine
    """
    output_var = input_var.get_transformed_path_and_affine()

def matplotlib_artist_TransformedPath_get_transformed_points_and_affine(self: 'any') -> 'any_computed':
    r"""
    Return a copy of the child path, with the non-affine part of
    the transform already applied, along with the affine part of
    the path necessary to complete the transformation.  Unlike
    :meth:`get_transformed_path_and_affine`, no interpolation will
    be performed.
    Keywords: matplotlib.artist.TransformedPath, get_transformed_points_and_affine
    """
    output_var = input_var.get_transformed_points_and_affine()

def matplotlib_artist_TransformedPath_invalidate(self: 'any') -> 'any_computed':
    r"""
    Invalidate this `TransformNode` and triggers an invalidation of its
    ancestors.  Should be called any time the transform changes.
    Keywords: matplotlib.artist.TransformedPath, invalidate
    """
    output_var = input_var.invalidate()

def matplotlib_artist_TransformedPath_set_children(self: 'any') -> 'any_computed':
    r"""
    Set the children of the transform, to let the invalidation
    system know which transforms can invalidate this transform.
    Should be called from the constructor of any transforms that
    depend on other transforms.
    Keywords: matplotlib.artist.TransformedPath, set_children
    """
    output_var = input_var.set_children()

def matplotlib_artist_allow_rasterization(draw: 'any') -> 'any_computed':
    r"""
    Decorator for Artist.draw method. Provides routines
    that run before and after the draw call. The before and after functions
    are useful for changing artist-dependent renderer attributes or making
    other setup function calls, such as starting and flushing a mixed-mode
    renderer.
    Keywords: matplotlib.artist, allow_rasterization
    """
    output_var = matplotlib.artist.allow_rasterization(input_var)

def matplotlib_artist_cache(user_function: 'any') -> 'any_computed':
    r"""
    Simple lightweight unbounded cache.  Sometimes called "memoize".
    Keywords: matplotlib.artist, cache
    """
    output_var = matplotlib.artist.cache(input_var)

def matplotlib_cbook_CallbackRegistry_blocked(self: 'any') -> 'any_computed':
    r"""
    Block callback signals from being processed.
    A context manager to temporarily block/disable callback signals
    from being processed by the registered listeners.
    Keywords: matplotlib.cbook.CallbackRegistry, blocked
    """
    output_var = input_var.blocked()

def matplotlib_cbook_CallbackRegistry_connect(self: 'any') -> 'any_computed':
    r"""
    Register *func* to be called when signal *signal* is generated.
    Keywords: matplotlib.cbook.CallbackRegistry, connect
    """
    output_var = input_var.connect()

def matplotlib_cbook_CallbackRegistry_disconnect(self: 'any') -> 'any_computed':
    r"""
    Disconnect the callback registered with callback id *cid*.
    No error is raised if such a callback does not exist.
    Keywords: matplotlib.cbook.CallbackRegistry, disconnect
    """
    output_var = input_var.disconnect()

def matplotlib_cbook_CallbackRegistry_process(self: 'any') -> 'any_computed':
    r"""
    Process signal *s*.
    All of the functions registered to receive callbacks on *s* will be
    called with ``*args`` and ``**kwargs``.
    Keywords: matplotlib.cbook.CallbackRegistry, process
    """
    output_var = input_var.process()

def matplotlib_cbook_Grouper_get_siblings(self: 'any') -> 'any_computed':
    r"""
    Return all of the items joined with *a*, including itself.
    Keywords: matplotlib.cbook.Grouper, get_siblings
    """
    output_var = input_var.get_siblings()

def matplotlib_cbook_Grouper_join(self: 'any') -> 'any_computed':
    r"""
    Join given arguments into the same set.  Accepts one or more arguments.
    Keywords: matplotlib.cbook.Grouper, join
    """
    output_var = input_var.join()

def matplotlib_cbook_Grouper_joined(self: 'any') -> 'any_computed':
    r"""
    Return whether *a* and *b* are members of the same set.
    Keywords: matplotlib.cbook.Grouper, joined
    """
    output_var = input_var.joined()

def matplotlib_cbook_Grouper_remove(self: 'any') -> 'any_computed':
    r"""
    Remove *a* from the grouper, doing nothing if it is not there.
    Keywords: matplotlib.cbook.Grouper, remove
    """
    output_var = input_var.remove()

def matplotlib_cbook_GrouperView_get_siblings(self: 'any') -> 'any_computed':
    r"""
    Return all of the items joined with *a*, including itself.
    Keywords: matplotlib.cbook.GrouperView, get_siblings
    """
    output_var = input_var.get_siblings()

def matplotlib_cbook_GrouperView_joined(self: 'any') -> 'any_computed':
    r"""
    Return whether *a* and *b* are members of the same set.
    Keywords: matplotlib.cbook.GrouperView, joined
    """
    output_var = input_var.joined()

def matplotlib_cbook_boxplot_stats(X: 'any') -> 'any_computed':
    r"""
    Return a list of dictionaries of statistics used to draw a series of box
    and whisker plots using `~.Axes.bxp`.
    Parameters
    ----------
    Keywords: matplotlib.cbook, boxplot_stats
    """
    output_var = matplotlib.cbook.boxplot_stats(input_var)

def matplotlib_cbook_contiguous_regions(mask: 'any') -> 'any_computed':
    r"""
    Return a list of (ind0, ind1) such that ``mask[ind0:ind1].all()`` is
    True and we cover all such regions.
    Keywords: matplotlib.cbook, contiguous_regions
    """
    output_var = matplotlib.cbook.contiguous_regions(input_var)

def matplotlib_cbook_delete_masked_points(args: 'any') -> 'any_computed':
    r"""
    Find all masked and/or non-finite points in a set of arguments,
    and return the arguments with only the unmasked points remaining.
    Arguments can be in any of 5 categories:
    Keywords: matplotlib.cbook, delete_masked_points
    """
    output_var = matplotlib.cbook.delete_masked_points(input_var)

def matplotlib_cbook_file_requires_unicode(x: 'any') -> 'any_computed':
    r"""
    Return whether the given writable file-like object requires Unicode to be
    written to it.
    Keywords: matplotlib.cbook, file_requires_unicode
    """
    output_var = matplotlib.cbook.file_requires_unicode(input_var)

def matplotlib_cbook_flatten(seq: 'any') -> 'any_computed':
    r"""
    Return a generator of flattened nested containers.
    For example:
        >>> from matplotlib.cbook import flatten
    Keywords: matplotlib.cbook, flatten
    """
    output_var = matplotlib.cbook.flatten(input_var)

def matplotlib_cbook_get_sample_data(fname: 'any') -> 'any_computed':
    r"""
    Return a sample data file.  *fname* is a path relative to the
    :file:`mpl-data/sample_data` directory.  If *asfileobj* is `True`
    return a file object, otherwise just a file path.
    Sample data files are stored in the 'mpl-data/sample_data' directory within
    Keywords: matplotlib.cbook, get_sample_data
    """
    output_var = matplotlib.cbook.get_sample_data(input_var)

def matplotlib_cbook_index_of(y: 'any') -> 'any_computed':
    r"""
    A helper function to create reasonable x values for the given *y*.
    This is used for plotting (x, y) if x values are not explicitly given.
    First try ``y.index`` (assuming *y* is a `pandas.Series`), if that
    Keywords: matplotlib.cbook, index_of
    """
    output_var = matplotlib.cbook.index_of(input_var)

def matplotlib_cbook_is_math_text(s: 'any') -> 'any_computed':
    r"""
    Return whether the string *s* contains math expressions.
    This is done by checking whether *s* contains an even number of
    non-escaped dollar signs.
    Keywords: matplotlib.cbook, is_math_text
    """
    output_var = matplotlib.cbook.is_math_text(input_var)

def matplotlib_cbook_is_scalar_or_string(val: 'any') -> 'any_computed':
    r"""
    Return whether the given object is a scalar or string like.
    Keywords: matplotlib.cbook, is_scalar_or_string
    """
    output_var = matplotlib.cbook.is_scalar_or_string(input_var)

def matplotlib_cbook_is_writable_file_like(obj: 'any') -> 'any_computed':
    r"""
    Return whether *obj* looks like a file object with a *write* method.
    Keywords: matplotlib.cbook, is_writable_file_like
    """
    output_var = matplotlib.cbook.is_writable_file_like(input_var)

def matplotlib_cbook_normalize_kwargs(kw: 'any') -> 'any_computed':
    r"""
    Helper function to normalize kwarg inputs.
    Parameters
    ----------
    kw : dict or None
    Keywords: matplotlib.cbook, normalize_kwargs
    """
    output_var = matplotlib.cbook.normalize_kwargs(input_var)

def matplotlib_cbook_open_file_cm(path_or_file: 'any') -> 'any_computed':
    r"""
    Pass through file objects and context-manage path-likes.
    Keywords: matplotlib.cbook, open_file_cm
    """
    output_var = matplotlib.cbook.open_file_cm(input_var)

def matplotlib_cbook_print_cycles(objects: 'any') -> 'any_computed':
    r"""
    Print loops of cyclic references in the given *objects*.
    It is often useful to pass in ``gc.garbage`` to find the cycles that are
    preventing some objects from being garbage collected.
    Keywords: matplotlib.cbook, print_cycles
    """
    output_var = matplotlib.cbook.print_cycles(input_var)

def matplotlib_cbook_pts_to_midstep(x: 'any') -> 'any_computed':
    r"""
    Convert continuous line to mid-steps.
    Given a set of ``N`` points convert to ``2N`` points which when connected
    linearly give a step function which changes values at the middle of the
    intervals.
    Keywords: matplotlib.cbook, pts_to_midstep
    """
    output_var = matplotlib.cbook.pts_to_midstep(input_var)

def matplotlib_cbook_pts_to_poststep(x: 'any') -> 'any_computed':
    r"""
    Convert continuous line to post-steps.
    Given a set of ``N`` points convert to ``2N + 1`` points, which when
    connected linearly give a step function which changes values at the end of
    the intervals.
    Keywords: matplotlib.cbook, pts_to_poststep
    """
    output_var = matplotlib.cbook.pts_to_poststep(input_var)

def matplotlib_cbook_pts_to_prestep(x: 'any') -> 'any_computed':
    r"""
    Convert continuous line to pre-steps.
    Given a set of ``N`` points, convert to ``2N - 1`` points, which when
    connected linearly give a step function which changes values at the
    beginning of the intervals.
    Keywords: matplotlib.cbook, pts_to_prestep
    """
    output_var = matplotlib.cbook.pts_to_prestep(input_var)

def matplotlib_cbook_safe_first_element(obj: 'any') -> 'any_computed':
    r"""
    Return the first element in *obj*.
    This is a type-independent way of obtaining the first element,
    supporting both index access and the iterator protocol.
    Keywords: matplotlib.cbook, safe_first_element
    """
    output_var = matplotlib.cbook.safe_first_element(input_var)

def matplotlib_cbook_safe_masked_invalid(x: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.cbook, safe_masked_invalid
    """
    output_var = matplotlib.cbook.safe_masked_invalid(input_var)

def matplotlib_cbook_sanitize_sequence(data: 'any') -> 'any_computed':
    r"""
    Convert dictview objects to list. Other inputs are returned unchanged.
    Keywords: matplotlib.cbook, sanitize_sequence
    """
    output_var = matplotlib.cbook.sanitize_sequence(input_var)

def matplotlib_cbook_silent_list_append(self: 'any') -> 'any_computed':
    r"""
    Append object to the end of the list.
    Keywords: matplotlib.cbook.silent_list, append
    """
    output_var = input_var.append()

def matplotlib_cbook_silent_list_clear(self: 'any') -> 'any_computed':
    r"""
    Remove all items from list.
    Keywords: matplotlib.cbook.silent_list, clear
    """
    output_var = input_var.clear()

def matplotlib_cbook_silent_list_copy(self: 'any') -> 'any_computed':
    r"""
    Return a shallow copy of the list.
    Keywords: matplotlib.cbook.silent_list, copy
    """
    output_var = input_var.copy()

def matplotlib_cbook_silent_list_count(self: 'any') -> 'any_computed':
    r"""
    Return number of occurrences of value.
    Keywords: matplotlib.cbook.silent_list, count
    """
    output_var = input_var.count()

def matplotlib_cbook_silent_list_extend(self: 'any') -> 'any_computed':
    r"""
    Extend list by appending elements from the iterable.
    Keywords: matplotlib.cbook.silent_list, extend
    """
    output_var = input_var.extend()

def matplotlib_cbook_silent_list_index(self: 'any') -> 'any_computed':
    r"""
    Return first index of value.
    Raises ValueError if the value is not present.
    Keywords: matplotlib.cbook.silent_list, index
    """
    output_var = input_var.index()

def matplotlib_cbook_silent_list_insert(self: 'any') -> 'any_computed':
    r"""
    Insert object before index.
    Keywords: matplotlib.cbook.silent_list, insert
    """
    output_var = input_var.insert()

def matplotlib_cbook_silent_list_pop(self: 'any') -> 'any_computed':
    r"""
    Remove and return item at index (default last).
    Raises IndexError if list is empty or index is out of range.
    Keywords: matplotlib.cbook.silent_list, pop
    """
    output_var = input_var.pop()

def matplotlib_cbook_silent_list_remove(self: 'any') -> 'any_computed':
    r"""
    Remove first occurrence of value.
    Raises ValueError if the value is not present.
    Keywords: matplotlib.cbook.silent_list, remove
    """
    output_var = input_var.remove()

def matplotlib_cbook_silent_list_reverse(self: 'any') -> 'any_computed':
    r"""
    Reverse *IN PLACE*.
    Keywords: matplotlib.cbook.silent_list, reverse
    """
    output_var = input_var.reverse()

def matplotlib_cbook_silent_list_sort(self: 'any') -> 'any_computed':
    r"""
    Sort the list in ascending order and return None.
    The sort is in-place (i.e. the list itself is modified) and stable (i.e. the
    order of two equal elements is maintained).
    Keywords: matplotlib.cbook.silent_list, sort
    """
    output_var = input_var.sort()

def matplotlib_cbook_simple_linear_interpolation(a: 'any') -> 'any_computed':
    r"""
    Resample an array with ``steps - 1`` points between original point pairs.
    Along each column of *a*, ``(steps - 1)`` points are introduced between
    each original values; the values are linearly interpolated.
    Keywords: matplotlib.cbook, simple_linear_interpolation
    """
    output_var = matplotlib.cbook.simple_linear_interpolation(input_var)

def matplotlib_cbook_strip_math(s: 'any') -> 'any_computed':
    r"""
    Remove latex formatting from mathtext.
    Only handles fully math and fully non-math strings.
    Keywords: matplotlib.cbook, strip_math
    """
    output_var = matplotlib.cbook.strip_math(input_var)

def matplotlib_cbook_to_filehandle(fname: 'any') -> 'any_computed':
    r"""
    Convert a path to an open file handle or pass-through a file-like object.
    Consider using `open_file_cm` instead, as it allows one to properly close
    newly created file objects more easily.
    Keywords: matplotlib.cbook, to_filehandle
    """
    output_var = matplotlib.cbook.to_filehandle(input_var)

def matplotlib_cbook_violin_stats(X: 'any') -> 'any_computed':
    r"""
    Return a list of dictionaries of data which can be used to draw a series
    of violin plots.
    See the ``Returns`` section below to view the required keys of the
    dictionary.
    Keywords: matplotlib.cbook, violin_stats
    """
    output_var = matplotlib.cbook.violin_stats(input_var)

def matplotlib_artist_get(obj: 'any') -> 'any_computed':
    r"""
    Return the value of an `.Artist`'s *property*, or print all of them.
    Parameters
    ----------
    obj : `~matplotlib.artist.Artist`
    Keywords: matplotlib.artist, get
    """
    output_var = matplotlib.artist.get(input_var)

def matplotlib_artist_getp(obj: 'any') -> 'any_computed':
    r"""
    Return the value of an `.Artist`'s *property*, or print all of them.
    Parameters
    ----------
    obj : `~matplotlib.artist.Artist`
    Keywords: matplotlib.artist, getp
    """
    output_var = matplotlib.artist.getp(input_var)

def matplotlib_artist_kwdoc(artist: 'any') -> 'any_computed':
    r"""
    Inspect an `~matplotlib.artist.Artist` class (using `.ArtistInspector`) and
    return information about its settable properties and their current values.
    Parameters
    ----------
    Keywords: matplotlib.artist, kwdoc
    """
    output_var = matplotlib.artist.kwdoc(input_var)

def matplotlib_artist_namedtuple(typename: 'any') -> 'any_computed':
    r"""
    Returns a new subclass of tuple with named fields.
    >>> Point = namedtuple('Point', ['x', 'y'])
    >>> Point.__doc__                   # docstring for the new class
    'Point(x, y)'
    Keywords: matplotlib.artist, namedtuple
    """
    output_var = matplotlib.artist.namedtuple(input_var)

def matplotlib_artist_setp(obj: 'any') -> 'any_computed':
    r"""
    Set one or more properties on an `.Artist`, or list allowed values.
    Parameters
    ----------
    obj : `~matplotlib.artist.Artist` or list of `.Artist`
    Keywords: matplotlib.artist, setp
    """
    output_var = matplotlib.artist.setp(input_var)

def matplotlib_artist_wraps(wrapped: 'any') -> 'any_computed':
    r"""
    Decorator factory to apply update_wrapper() to a wrapper function
    Returns a decorator that invokes update_wrapper() with the decorated
    function as the wrapper argument and the arguments to wraps() as the
    remaining arguments. Default arguments are as for update_wrapper().
    Keywords: matplotlib.artist, wraps
    """
    output_var = matplotlib.artist.wraps(input_var)

def matplotlib_backends_registry_BackendRegistry_backend_for_gui_framework(self: 'any') -> 'any_computed':
    r"""
    Return the name of the backend corresponding to the specified GUI framework.
    Parameters
    ----------
    framework : str
    Keywords: matplotlib.backends.registry.BackendRegistry, backend_for_gui_framework
    """
    output_var = input_var.backend_for_gui_framework()

def matplotlib_backends_registry_BackendRegistry_is_valid_backend(self: 'any') -> 'any_computed':
    r"""
    Return True if the backend name is valid, False otherwise.
    A backend name is valid if it is one of the built-in backends or has been
    dynamically added via an entry point. Those beginning with ``module://`` are
    always considered valid and are added to the current list of all backends
    Keywords: matplotlib.backends.registry.BackendRegistry, is_valid_backend
    """
    output_var = input_var.is_valid_backend()

def matplotlib_backends_registry_BackendRegistry_list_all(self: 'any') -> 'any_computed':
    r"""
    Return list of all known backends.
    These include built-in backends and those obtained at runtime either from entry
    points or explicit ``module://some.backend`` syntax.
    Keywords: matplotlib.backends.registry.BackendRegistry, list_all
    """
    output_var = input_var.list_all()

def matplotlib_backends_registry_BackendRegistry_list_builtin(self: 'any') -> 'any_computed':
    r"""
    Return list of backends that are built into Matplotlib.
    Parameters
    ----------
    filter_ : `~.BackendFilter`, optional
    Keywords: matplotlib.backends.registry.BackendRegistry, list_builtin
    """
    output_var = input_var.list_builtin()

def matplotlib_backends_registry_BackendRegistry_list_gui_frameworks(self: 'any') -> 'any_computed':
    r"""
    Return list of GUI frameworks used by Matplotlib backends.
    Returns
    -------
    list of str
    Keywords: matplotlib.backends.registry.BackendRegistry, list_gui_frameworks
    """
    output_var = input_var.list_gui_frameworks()

def matplotlib_backends_registry_BackendRegistry_load_backend_module(self: 'any') -> 'any_computed':
    r"""
    Load and return the module containing the specified backend.
    Parameters
    ----------
    backend : str
    Keywords: matplotlib.backends.registry.BackendRegistry, load_backend_module
    """
    output_var = input_var.load_backend_module()

def matplotlib_backends_registry_BackendRegistry_resolve_backend(self: 'any') -> 'any_computed':
    r"""
    Return the backend and GUI framework for the specified backend name.
    If the GUI framework is not yet known then it will be determined by loading the
    backend module and checking the ``FigureCanvas.required_interactive_framework``
    attribute.
    Keywords: matplotlib.backends.registry.BackendRegistry, resolve_backend
    """
    output_var = input_var.resolve_backend()

def matplotlib_backends_registry_BackendRegistry_resolve_gui_or_backend(self: 'any') -> 'any_computed':
    r"""
    Return the backend and GUI framework for the specified string that may be
    either a GUI framework or a backend name, tested in that order.
    This is for use with the IPython %matplotlib magic command which may be a GUI
    framework such as ``%matplotlib qt`` or a backend name such as
    Keywords: matplotlib.backends.registry.BackendRegistry, resolve_gui_or_backend
    """
    output_var = input_var.resolve_gui_or_backend()

def matplotlib_bezier_BezierSegment_axis_aligned_extrema(self: 'any') -> 'any_computed':
    r"""
    Return the dimension and location of the curve's interior extrema.
    The extrema are the points along the curve where one of its partial
    derivatives is zero.
    Keywords: matplotlib.bezier.BezierSegment, axis_aligned_extrema
    """
    output_var = input_var.axis_aligned_extrema()

def matplotlib_bezier_BezierSegment_point_at_t(self: 'any') -> 'any_computed':
    r"""
    Evaluate the curve at a single point, returning a tuple of *d* floats.
    Keywords: matplotlib.bezier.BezierSegment, point_at_t
    """
    output_var = input_var.point_at_t()

def matplotlib_bezier_NonIntersectingPathException_add_note(self: 'any') -> 'any_computed':
    r"""
    Exception.add_note(note) --
    add a note to the exception
    Keywords: matplotlib.bezier.NonIntersectingPathException, add_note
    """
    output_var = input_var.add_note()

def matplotlib_bezier_NonIntersectingPathException_with_traceback(self: 'any') -> 'any_computed':
    r"""
    Exception.with_traceback(tb) --
    set self.__traceback__ to tb and return self.
    Keywords: matplotlib.bezier.NonIntersectingPathException, with_traceback
    """
    output_var = input_var.with_traceback()

def matplotlib_bezier_check_if_parallel(dx1: 'any') -> 'any_computed':
    r"""
    Check if two lines are parallel.
    Parameters
    ----------
    dx1, dy1, dx2, dy2 : float
    Keywords: matplotlib.bezier, check_if_parallel
    """
    output_var = matplotlib.bezier.check_if_parallel(input_var)

def matplotlib_bezier_find_bezier_t_intersecting_with_closedpath(bezier_point_at_t: 'any') -> 'any_computed':
    r"""
    Find the intersection of the Bézier curve with a closed path.
    The intersection point *t* is approximated by two parameters *t0*, *t1*
    such that *t0* <= *t* <= *t1*.
    Keywords: matplotlib.bezier, find_bezier_t_intersecting_with_closedpath
    """
    output_var = matplotlib.bezier.find_bezier_t_intersecting_with_closedpath(input_var)

def matplotlib_bezier_find_control_points(c1x: 'any') -> 'any_computed':
    r"""
    Find control points of the Bézier curve passing through (*c1x*, *c1y*),
    (*mmx*, *mmy*), and (*c2x*, *c2y*), at parametric values 0, 0.5, and 1.
    Keywords: matplotlib.bezier, find_control_points
    """
    output_var = matplotlib.bezier.find_control_points(input_var)

def matplotlib_bezier_get_cos_sin(x0: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.bezier, get_cos_sin
    """
    output_var = matplotlib.bezier.get_cos_sin(input_var)

def matplotlib_bezier_get_intersection(cx1: 'any') -> 'any_computed':
    r"""
    Return the intersection between the line through (*cx1*, *cy1*) at angle
    *t1* and the line through (*cx2*, *cy2*) at angle *t2*.
    Keywords: matplotlib.bezier, get_intersection
    """
    output_var = matplotlib.bezier.get_intersection(input_var)

def matplotlib_bezier_get_normal_points(cx: 'any') -> 'any_computed':
    r"""
    For a line passing through (*cx*, *cy*) and having an angle *t*, return
    locations of the two points located along its perpendicular line at the
    distance of *length*.
    Keywords: matplotlib.bezier, get_normal_points
    """
    output_var = matplotlib.bezier.get_normal_points(input_var)

def matplotlib_bezier_get_parallels(bezier2: 'any') -> 'any_computed':
    r"""
    Given the quadratic Bézier control points *bezier2*, returns
    control points of quadratic Bézier lines roughly parallel to given
    one separated by *width*.
    Keywords: matplotlib.bezier, get_parallels
    """
    output_var = matplotlib.bezier.get_parallels(input_var)

def matplotlib_bezier_inside_circle(cx: 'any') -> 'any_computed':
    r"""
    Return a function that checks whether a point is in a circle with center
    (*cx*, *cy*) and radius *r*.
    The returned function has the signature::
    Keywords: matplotlib.bezier, inside_circle
    """
    output_var = matplotlib.bezier.inside_circle(input_var)

def matplotlib_bezier_lru_cache(maxsize: 'any') -> 'any_computed':
    r"""
    Least-recently-used cache decorator.
    If *maxsize* is set to None, the LRU features are disabled and the cache
    can grow without bound.
    Keywords: matplotlib.bezier, lru_cache
    """
    output_var = matplotlib.bezier.lru_cache(input_var)

def matplotlib_bezier_make_wedged_bezier2(bezier2: 'any') -> 'any_computed':
    r"""
    Being similar to `get_parallels`, returns control points of two quadratic
    Bézier lines having a width roughly parallel to given one separated by
    *width*.
    Keywords: matplotlib.bezier, make_wedged_bezier2
    """
    output_var = matplotlib.bezier.make_wedged_bezier2(input_var)

def matplotlib_bezier_split_bezier_intersecting_with_closedpath(bezier: 'any') -> 'any_computed':
    r"""
    Split a Bézier curve into two at the intersection with a closed path.
    Parameters
    ----------
    bezier : (N, 2) array-like
    Keywords: matplotlib.bezier, split_bezier_intersecting_with_closedpath
    """
    output_var = matplotlib.bezier.split_bezier_intersecting_with_closedpath(input_var)

def matplotlib_bezier_split_de_casteljau(beta: 'any') -> 'any_computed':
    r"""
    Split a Bézier segment defined by its control points *beta* into two
    separate segments divided at *t* and return their control points.
    Keywords: matplotlib.bezier, split_de_casteljau
    """
    output_var = matplotlib.bezier.split_de_casteljau(input_var)

def matplotlib_bezier_split_path_inout(path: 'any') -> 'any_computed':
    r"""
    Divide a path into two segments at the point where ``inside(x, y)`` becomes
    False.
    Keywords: matplotlib.bezier, split_path_inout
    """
    output_var = matplotlib.bezier.split_path_inout(input_var)

def matplotlib_cm_ColormapRegistry_get(self: 'any') -> 'any_computed':
    r"""
    D.get(k[,d]) -> D[k] if k in D, else d.  d defaults to None.
    Keywords: matplotlib.cm.ColormapRegistry, get
    """
    output_var = input_var.get()

def matplotlib_cm_ColormapRegistry_get_cmap(self: 'any') -> 'any_computed':
    r"""
    Return a color map specified through *cmap*.
    Parameters
    ----------
    cmap : str or `~matplotlib.colors.Colormap` or None
    Keywords: matplotlib.cm.ColormapRegistry, get_cmap
    """
    output_var = input_var.get_cmap()

def matplotlib_cm_ColormapRegistry_items(self: 'any') -> 'any_computed':
    r"""
    D.items() -> a set-like object providing a view on D's items
    Keywords: matplotlib.cm.ColormapRegistry, items
    """
    output_var = input_var.items()

def matplotlib_cm_ColormapRegistry_keys(self: 'any') -> 'any_computed':
    r"""
    D.keys() -> a set-like object providing a view on D's keys
    Keywords: matplotlib.cm.ColormapRegistry, keys
    """
    output_var = input_var.keys()

def matplotlib_cm_ColormapRegistry_register(self: 'any') -> 'any_computed':
    r"""
    Register a new colormap.
    The colormap name can then be used as a string argument to any ``cmap``
    parameter in Matplotlib. It is also available in ``pyplot.get_cmap``.
    Keywords: matplotlib.cm.ColormapRegistry, register
    """
    output_var = input_var.register()

def matplotlib_cm_ColormapRegistry_unregister(self: 'any') -> 'any_computed':
    r"""
    Remove a colormap from the registry.
    You cannot remove built-in colormaps.
    If the named colormap is not registered, returns with no error, raises
    Keywords: matplotlib.cm.ColormapRegistry, unregister
    """
    output_var = input_var.unregister()

def matplotlib_cm_ColormapRegistry_values(self: 'any') -> 'any_computed':
    r"""
    D.values() -> an object providing a view on D's values
    Keywords: matplotlib.cm.ColormapRegistry, values
    """
    output_var = input_var.values()

def matplotlib_cm_ScalarMappable_autoscale(self: 'any') -> 'any_computed':
    r"""
    Autoscale the scalar limits on the norm instance using the
    current array
    Keywords: matplotlib.cm.ScalarMappable, autoscale
    """
    output_var = input_var.autoscale()

def matplotlib_cm_ScalarMappable_autoscale_None(self: 'any') -> 'any_computed':
    r"""
    Autoscale the scalar limits on the norm instance using the
    current array, changing only limits that are None
    Keywords: matplotlib.cm.ScalarMappable, autoscale_None
    """
    output_var = input_var.autoscale_None()

def matplotlib_cm_ScalarMappable_changed(self: 'any') -> 'any_computed':
    r"""
    Call this whenever the mappable is changed to notify all the
    callbackSM listeners to the 'changed' signal.
    Keywords: matplotlib.cm.ScalarMappable, changed
    """
    output_var = input_var.changed()

def matplotlib_cm_ScalarMappable_get_alpha(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.cm.ScalarMappable, get_alpha
    """
    output_var = input_var.get_alpha()

def matplotlib_cm_ScalarMappable_get_array(self: 'any') -> 'any_computed':
    r"""
    Return the array of values, that are mapped to colors.
    The base class `.ScalarMappable` does not make any assumptions on
    the dimensionality and shape of the array.
    Keywords: matplotlib.cm.ScalarMappable, get_array
    """
    output_var = input_var.get_array()

def matplotlib_cm_ScalarMappable_get_clim(self: 'any') -> 'any_computed':
    r"""
    Return the values (min, max) that are mapped to the colormap limits.
    Keywords: matplotlib.cm.ScalarMappable, get_clim
    """
    output_var = input_var.get_clim()

def matplotlib_cm_ScalarMappable_get_cmap(self: 'any') -> 'any_computed':
    r"""
    Return the `.Colormap` instance.
    Keywords: matplotlib.cm.ScalarMappable, get_cmap
    """
    output_var = input_var.get_cmap()

def matplotlib_cm_ScalarMappable_set_array(self: 'any') -> 'any_computed':
    r"""
    Set the value array from array-like *A*.
    Parameters
    ----------
    A : array-like or None
    Keywords: matplotlib.cm.ScalarMappable, set_array
    """
    output_var = input_var.set_array()

def matplotlib_cm_ScalarMappable_set_clim(self: 'any') -> 'any_computed':
    r"""
    Set the norm limits for image scaling.
    Parameters
    ----------
    vmin, vmax : float
    Keywords: matplotlib.cm.ScalarMappable, set_clim
    """
    output_var = input_var.set_clim()

def matplotlib_cm_ScalarMappable_set_cmap(self: 'any') -> 'any_computed':
    r"""
    Set the colormap for luminance data.
    Parameters
    ----------
    cmap : `.Colormap` or str or None
    Keywords: matplotlib.cm.ScalarMappable, set_cmap
    """
    output_var = input_var.set_cmap()

def matplotlib_cm_ScalarMappable_set_norm(self: 'any') -> 'any_computed':
    r"""
    Set the normalization instance.
    Parameters
    ----------
    norm : `.Normalize` or str or None
    Keywords: matplotlib.cm.ScalarMappable, set_norm
    """
    output_var = input_var.set_norm()

def matplotlib_cm_ScalarMappable_to_rgba(self: 'any') -> 'any_computed':
    r"""
    Return a normalized RGBA array corresponding to *x*.
    In the normal case, *x* is a 1D or 2D sequence of scalars, and
    the corresponding `~numpy.ndarray` of RGBA values will be returned,
    based on the norm and colormap set for this Colorizer.
    Keywords: matplotlib.cm.ScalarMappable, to_rgba
    """
    output_var = input_var.to_rgba()

def matplotlib_colors_AsinhNorm_autoscale(self: 'any') -> 'any_computed':
    r"""
    Set *vmin*, *vmax* to min, max of *A*.
    Keywords: matplotlib.colors.AsinhNorm, autoscale
    """
    output_var = input_var.autoscale()

def matplotlib_colors_AsinhNorm_autoscale_None(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.colors.AsinhNorm, autoscale_None
    """
    output_var = input_var.autoscale_None()

def matplotlib_colors_AsinhNorm_inverse(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.colors.AsinhNorm, inverse
    """
    output_var = input_var.inverse()

def matplotlib_colors_AsinhNorm_process_value(value: 'any') -> 'any_computed':
    r"""
    Homogenize the input *value* for easy and efficient normalization.
    *value* can be a scalar or sequence.
    Parameters
    Keywords: matplotlib.colors.AsinhNorm, process_value
    """
    output_var = input_var.process_value()

def matplotlib_colors_AsinhNorm_scaled(self: 'any') -> 'any_computed':
    r"""
    Return whether *vmin* and *vmax* are both set.
    Keywords: matplotlib.colors.AsinhNorm, scaled
    """
    output_var = input_var.scaled()

def matplotlib_colors_BivarColormap_copy(self: 'any') -> 'any_computed':
    r"""
    Return a copy of the colormap.
    Keywords: matplotlib.colors.BivarColormap, copy
    """
    output_var = input_var.copy()

def matplotlib_colors_BivarColormap_get_bad(self: 'any') -> 'any_computed':
    r"""
    Get the color for masked values.
    Keywords: matplotlib.colors.BivarColormap, get_bad
    """
    output_var = input_var.get_bad()

def matplotlib_colors_BivarColormap_get_outside(self: 'any') -> 'any_computed':
    r"""
    Get the color for out-of-range values.
    Keywords: matplotlib.colors.BivarColormap, get_outside
    """
    output_var = input_var.get_outside()

def matplotlib_colors_BivarColormap_resampled(self: 'any') -> 'any_computed':
    r"""
    Return a new colormap with *lutshape* entries.
    Note that this function does not move the origin.
    Parameters
    Keywords: matplotlib.colors.BivarColormap, resampled
    """
    output_var = input_var.resampled()

def matplotlib_colors_BivarColormap_reversed(self: 'any') -> 'any_computed':
    r"""
    Reverses both or one of the axis.
    Keywords: matplotlib.colors.BivarColormap, reversed
    """
    output_var = input_var.reversed()

def matplotlib_colors_BivarColormap_transposed(self: 'any') -> 'any_computed':
    r"""
    Transposes the colormap by swapping the order of the axis
    Keywords: matplotlib.colors.BivarColormap, transposed
    """
    output_var = input_var.transposed()

def matplotlib_colors_BivarColormap_with_extremes(self: 'any') -> 'any_computed':
    r"""
    Return a copy of the `BivarColormap` with modified attributes.
    Note that the *outside* color is only relevant if `shape` = 'ignore'
    or 'circleignore'.
    Keywords: matplotlib.colors.BivarColormap, with_extremes
    """
    output_var = input_var.with_extremes()

def matplotlib_colors_BivarColormapFromImage_copy(self: 'any') -> 'any_computed':
    r"""
    Return a copy of the colormap.
    Keywords: matplotlib.colors.BivarColormapFromImage, copy
    """
    output_var = input_var.copy()

def matplotlib_colors_BivarColormapFromImage_get_bad(self: 'any') -> 'any_computed':
    r"""
    Get the color for masked values.
    Keywords: matplotlib.colors.BivarColormapFromImage, get_bad
    """
    output_var = input_var.get_bad()

def matplotlib_colors_BivarColormapFromImage_get_outside(self: 'any') -> 'any_computed':
    r"""
    Get the color for out-of-range values.
    Keywords: matplotlib.colors.BivarColormapFromImage, get_outside
    """
    output_var = input_var.get_outside()

def matplotlib_colors_BivarColormapFromImage_resampled(self: 'any') -> 'any_computed':
    r"""
    Return a new colormap with *lutshape* entries.
    Note that this function does not move the origin.
    Parameters
    Keywords: matplotlib.colors.BivarColormapFromImage, resampled
    """
    output_var = input_var.resampled()

def matplotlib_colors_BivarColormapFromImage_reversed(self: 'any') -> 'any_computed':
    r"""
    Reverses both or one of the axis.
    Keywords: matplotlib.colors.BivarColormapFromImage, reversed
    """
    output_var = input_var.reversed()

def matplotlib_colors_BivarColormapFromImage_transposed(self: 'any') -> 'any_computed':
    r"""
    Transposes the colormap by swapping the order of the axis
    Keywords: matplotlib.colors.BivarColormapFromImage, transposed
    """
    output_var = input_var.transposed()

def matplotlib_colors_BivarColormapFromImage_with_extremes(self: 'any') -> 'any_computed':
    r"""
    Return a copy of the `BivarColormap` with modified attributes.
    Note that the *outside* color is only relevant if `shape` = 'ignore'
    or 'circleignore'.
    Keywords: matplotlib.colors.BivarColormapFromImage, with_extremes
    """
    output_var = input_var.with_extremes()

def matplotlib_colors_BoundaryNorm_autoscale(self: 'any') -> 'any_computed':
    r"""
    Set *vmin*, *vmax* to min, max of *A*.
    Keywords: matplotlib.colors.BoundaryNorm, autoscale
    """
    output_var = input_var.autoscale()

def matplotlib_colors_BoundaryNorm_autoscale_None(self: 'any') -> 'any_computed':
    r"""
    If *vmin* or *vmax* are not set, use the min/max of *A* to set them.
    Keywords: matplotlib.colors.BoundaryNorm, autoscale_None
    """
    output_var = input_var.autoscale_None()

def matplotlib_colors_BoundaryNorm_inverse(self: 'any') -> 'any_computed':
    r"""
    Raises
    ------
    ValueError
        BoundaryNorm is not invertible, so calling this method will always
        raise an error
    Keywords: matplotlib.colors.BoundaryNorm, inverse
    """
    output_var = input_var.inverse()

def matplotlib_colors_BoundaryNorm_process_value(value: 'any') -> 'any_computed':
    r"""
    Homogenize the input *value* for easy and efficient normalization.
    *value* can be a scalar or sequence.
    Parameters
    Keywords: matplotlib.colors.BoundaryNorm, process_value
    """
    output_var = input_var.process_value()

def matplotlib_colors_BoundaryNorm_scaled(self: 'any') -> 'any_computed':
    r"""
    Return whether *vmin* and *vmax* are both set.
    Keywords: matplotlib.colors.BoundaryNorm, scaled
    """
    output_var = input_var.scaled()

def matplotlib_colors_CenteredNorm_autoscale(self: 'any') -> 'any_computed':
    r"""
    Set *halfrange* to ``max(abs(A-vcenter))``, then set *vmin* and *vmax*.
    Keywords: matplotlib.colors.CenteredNorm, autoscale
    """
    output_var = input_var.autoscale()

def matplotlib_colors_CenteredNorm_autoscale_None(self: 'any') -> 'any_computed':
    r"""
    Set *vmin* and *vmax*.
    Keywords: matplotlib.colors.CenteredNorm, autoscale_None
    """
    output_var = input_var.autoscale_None()

def matplotlib_colors_CenteredNorm_inverse(self: 'any') -> 'any_computed':
    r"""
    Maps the normalized value (i.e., index in the colormap) back to image
    data value.
    Parameters
    ----------
    Keywords: matplotlib.colors.CenteredNorm, inverse
    """
    output_var = input_var.inverse()

def matplotlib_colors_CenteredNorm_process_value(value: 'any') -> 'any_computed':
    r"""
    Homogenize the input *value* for easy and efficient normalization.
    *value* can be a scalar or sequence.
    Parameters
    Keywords: matplotlib.colors.CenteredNorm, process_value
    """
    output_var = input_var.process_value()

def matplotlib_colors_CenteredNorm_scaled(self: 'any') -> 'any_computed':
    r"""
    Return whether *vmin* and *vmax* are both set.
    Keywords: matplotlib.colors.CenteredNorm, scaled
    """
    output_var = input_var.scaled()

def matplotlib_colors_ColorConverter_to_rgb(c: 'any') -> 'any_computed':
    r"""
    Convert *c* to an RGB color, silently dropping the alpha channel.
    Keywords: matplotlib.colors.ColorConverter, to_rgb
    """
    output_var = input_var.to_rgb()

def matplotlib_colors_ColorConverter_to_rgba(c: 'any') -> 'any_computed':
    r"""
    Convert *c* to an RGBA color.
    Parameters
    ----------
    c : Matplotlib color or ``np.ma.masked``
    Keywords: matplotlib.colors.ColorConverter, to_rgba
    """
    output_var = input_var.to_rgba()

def matplotlib_colors_ColorConverter_to_rgba_array(c: 'any') -> 'any_computed':
    r"""
    Convert *c* to a (n, 4) array of RGBA colors.
    Parameters
    ----------
    c : Matplotlib color or array of colors
    Keywords: matplotlib.colors.ColorConverter, to_rgba_array
    """
    output_var = input_var.to_rgba_array()

def matplotlib_colors_ColorSequenceRegistry_get(self: 'any') -> 'any_computed':
    r"""
    D.get(k[,d]) -> D[k] if k in D, else d.  d defaults to None.
    Keywords: matplotlib.colors.ColorSequenceRegistry, get
    """
    output_var = input_var.get()

def matplotlib_colors_ColorSequenceRegistry_items(self: 'any') -> 'any_computed':
    r"""
    D.items() -> a set-like object providing a view on D's items
    Keywords: matplotlib.colors.ColorSequenceRegistry, items
    """
    output_var = input_var.items()

def matplotlib_colors_ColorSequenceRegistry_keys(self: 'any') -> 'any_computed':
    r"""
    D.keys() -> a set-like object providing a view on D's keys
    Keywords: matplotlib.colors.ColorSequenceRegistry, keys
    """
    output_var = input_var.keys()

def matplotlib_colors_ColorSequenceRegistry_register(self: 'any') -> 'any_computed':
    r"""
    Register a new color sequence.
    The color sequence registry stores a copy of the given *color_list*, so
    that future changes to the original list do not affect the registered
    color sequence. Think of this as the registry taking a snapshot
    Keywords: matplotlib.colors.ColorSequenceRegistry, register
    """
    output_var = input_var.register()

def matplotlib_colors_ColorSequenceRegistry_unregister(self: 'any') -> 'any_computed':
    r"""
    Remove a sequence from the registry.
    You cannot remove built-in color sequences.
    If the name is not registered, returns with no error.
    Keywords: matplotlib.colors.ColorSequenceRegistry, unregister
    """
    output_var = input_var.unregister()

def matplotlib_colors_ColorSequenceRegistry_values(self: 'any') -> 'any_computed':
    r"""
    D.values() -> an object providing a view on D's values
    Keywords: matplotlib.colors.ColorSequenceRegistry, values
    """
    output_var = input_var.values()

def matplotlib_colors_Colormap_copy(self: 'any') -> 'any_computed':
    r"""
    Return a copy of the colormap.
    Keywords: matplotlib.colors.Colormap, copy
    """
    output_var = input_var.copy()

def matplotlib_colors_Colormap_get_bad(self: 'any') -> 'any_computed':
    r"""
    Get the color for masked values.
    Keywords: matplotlib.colors.Colormap, get_bad
    """
    output_var = input_var.get_bad()

def matplotlib_colors_Colormap_get_over(self: 'any') -> 'any_computed':
    r"""
    Get the color for high out-of-range values.
    Keywords: matplotlib.colors.Colormap, get_over
    """
    output_var = input_var.get_over()

def matplotlib_colors_Colormap_get_under(self: 'any') -> 'any_computed':
    r"""
    Get the color for low out-of-range values.
    Keywords: matplotlib.colors.Colormap, get_under
    """
    output_var = input_var.get_under()

def matplotlib_colors_Colormap_is_gray(self: 'any') -> 'any_computed':
    r"""
    Return whether the colormap is grayscale.
    Keywords: matplotlib.colors.Colormap, is_gray
    """
    output_var = input_var.is_gray()

def matplotlib_colors_Colormap_resampled(self: 'any') -> 'any_computed':
    r"""
    Return a new colormap with *lutsize* entries.
    Keywords: matplotlib.colors.Colormap, resampled
    """
    output_var = input_var.resampled()

def matplotlib_colors_Colormap_reversed(self: 'any') -> 'any_computed':
    r"""
    Return a reversed instance of the Colormap.
    .. note:: This function is not implemented for the base class.
    Parameters
    Keywords: matplotlib.colors.Colormap, reversed
    """
    output_var = input_var.reversed()

def matplotlib_colors_Colormap_set_bad(self: 'any') -> 'any_computed':
    r"""
    Set the color for masked values.
    Keywords: matplotlib.colors.Colormap, set_bad
    """
    output_var = input_var.set_bad()

def matplotlib_colors_Colormap_set_extremes(self: 'any') -> 'any_computed':
    r"""
    Set the colors for masked (*bad*) values and, when ``norm.clip =
    False``, low (*under*) and high (*over*) out-of-range values.
    Keywords: matplotlib.colors.Colormap, set_extremes
    """
    output_var = input_var.set_extremes()

def matplotlib_colors_Colormap_set_over(self: 'any') -> 'any_computed':
    r"""
    Set the color for high out-of-range values.
    Keywords: matplotlib.colors.Colormap, set_over
    """
    output_var = input_var.set_over()

def matplotlib_colors_Colormap_set_under(self: 'any') -> 'any_computed':
    r"""
    Set the color for low out-of-range values.
    Keywords: matplotlib.colors.Colormap, set_under
    """
    output_var = input_var.set_under()

def matplotlib_colors_Colormap_with_extremes(self: 'any') -> 'any_computed':
    r"""
    Return a copy of the colormap, for which the colors for masked (*bad*)
    values and, when ``norm.clip = False``, low (*under*) and high (*over*)
    out-of-range values, have been set accordingly.
    Keywords: matplotlib.colors.Colormap, with_extremes
    """
    output_var = input_var.with_extremes()

def matplotlib_colors_FuncNorm_autoscale(self: 'any') -> 'any_computed':
    r"""
    Set *vmin*, *vmax* to min, max of *A*.
    Keywords: matplotlib.colors.FuncNorm, autoscale
    """
    output_var = input_var.autoscale()

def matplotlib_colors_FuncNorm_autoscale_None(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.colors.FuncNorm, autoscale_None
    """
    output_var = input_var.autoscale_None()

def matplotlib_colors_FuncNorm_inverse(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.colors.FuncNorm, inverse
    """
    output_var = input_var.inverse()

def matplotlib_colors_FuncNorm_process_value(value: 'any') -> 'any_computed':
    r"""
    Homogenize the input *value* for easy and efficient normalization.
    *value* can be a scalar or sequence.
    Parameters
    Keywords: matplotlib.colors.FuncNorm, process_value
    """
    output_var = input_var.process_value()

def matplotlib_colors_FuncNorm_scaled(self: 'any') -> 'any_computed':
    r"""
    Return whether *vmin* and *vmax* are both set.
    Keywords: matplotlib.colors.FuncNorm, scaled
    """
    output_var = input_var.scaled()

def matplotlib_colors_LightSource_blend_hsv(self: 'any') -> 'any_computed':
    r"""
    Take the input data array, convert to HSV values in the given colormap,
    then adjust those color values to give the impression of a shaded
    relief map with a specified light source.  RGBA values are returned,
    which can then be used to plot the shaded image with imshow.
    Keywords: matplotlib.colors.LightSource, blend_hsv
    """
    output_var = input_var.blend_hsv()

def matplotlib_colors_LightSource_blend_overlay(self: 'any') -> 'any_computed':
    r"""
    Combine an RGB image with an intensity map using "overlay" blending.
    Parameters
    ----------
    rgb : `~numpy.ndarray`
    Keywords: matplotlib.colors.LightSource, blend_overlay
    """
    output_var = input_var.blend_overlay()

def matplotlib_colors_LightSource_blend_soft_light(self: 'any') -> 'any_computed':
    r"""
    Combine an RGB image with an intensity map using "soft light" blending,
    using the "pegtop" formula.
    Parameters
    ----------
    Keywords: matplotlib.colors.LightSource, blend_soft_light
    """
    output_var = input_var.blend_soft_light()

def matplotlib_colors_LightSource_hillshade(self: 'any') -> 'any_computed':
    r"""
    Calculate the illumination intensity for a surface using the defined
    azimuth and elevation for the light source.
    This computes the normal vectors for the surface, and then passes them
    on to `shade_normals`
    Keywords: matplotlib.colors.LightSource, hillshade
    """
    output_var = input_var.hillshade()

def matplotlib_colors_LightSource_shade(self: 'any') -> 'any_computed':
    r"""
    Combine colormapped data values with an illumination intensity map
    (a.k.a.  "hillshade") of the values.
    Parameters
    ----------
    Keywords: matplotlib.colors.LightSource, shade
    """
    output_var = input_var.shade()

def matplotlib_colors_LightSource_shade_normals(self: 'any') -> 'any_computed':
    r"""
    Calculate the illumination intensity for the normal vectors of a
    surface using the defined azimuth and elevation for the light source.
    Imagine an artificial sun placed at infinity in some azimuth and
    elevation position illuminating our surface. The parts of the surface
    Keywords: matplotlib.colors.LightSource, shade_normals
    """
    output_var = input_var.shade_normals()

def matplotlib_colors_LightSource_shade_rgb(self: 'any') -> 'any_computed':
    r"""
    Use this light source to adjust the colors of the *rgb* input array to
    give the impression of a shaded relief map with the given *elevation*.
    Parameters
    ----------
    Keywords: matplotlib.colors.LightSource, shade_rgb
    """
    output_var = input_var.shade_rgb()

def matplotlib_colors_LinearSegmentedColormap_copy(self: 'any') -> 'any_computed':
    r"""
    Return a copy of the colormap.
    Keywords: matplotlib.colors.LinearSegmentedColormap, copy
    """
    output_var = input_var.copy()

def matplotlib_colors_LinearSegmentedColormap_from_list(name: 'any') -> 'any_computed':
    r"""
    Create a `LinearSegmentedColormap` from a list of colors.
    Parameters
    ----------
    name : str
    Keywords: matplotlib.colors.LinearSegmentedColormap, from_list
    """
    output_var = input_var.from_list()

def matplotlib_colors_LinearSegmentedColormap_get_bad(self: 'any') -> 'any_computed':
    r"""
    Get the color for masked values.
    Keywords: matplotlib.colors.LinearSegmentedColormap, get_bad
    """
    output_var = input_var.get_bad()

def matplotlib_colors_LinearSegmentedColormap_get_over(self: 'any') -> 'any_computed':
    r"""
    Get the color for high out-of-range values.
    Keywords: matplotlib.colors.LinearSegmentedColormap, get_over
    """
    output_var = input_var.get_over()

def matplotlib_colors_LinearSegmentedColormap_get_under(self: 'any') -> 'any_computed':
    r"""
    Get the color for low out-of-range values.
    Keywords: matplotlib.colors.LinearSegmentedColormap, get_under
    """
    output_var = input_var.get_under()

def matplotlib_colors_LinearSegmentedColormap_is_gray(self: 'any') -> 'any_computed':
    r"""
    Return whether the colormap is grayscale.
    Keywords: matplotlib.colors.LinearSegmentedColormap, is_gray
    """
    output_var = input_var.is_gray()

def matplotlib_colors_LinearSegmentedColormap_resampled(self: 'any') -> 'any_computed':
    r"""
    Return a new colormap with *lutsize* entries.
    Keywords: matplotlib.colors.LinearSegmentedColormap, resampled
    """
    output_var = input_var.resampled()

def matplotlib_colors_LinearSegmentedColormap_reversed(self: 'any') -> 'any_computed':
    r"""
    Return a reversed instance of the Colormap.
    Parameters
    ----------
    name : str, optional
    Keywords: matplotlib.colors.LinearSegmentedColormap, reversed
    """
    output_var = input_var.reversed()

def matplotlib_colors_LinearSegmentedColormap_set_bad(self: 'any') -> 'any_computed':
    r"""
    Set the color for masked values.
    Keywords: matplotlib.colors.LinearSegmentedColormap, set_bad
    """
    output_var = input_var.set_bad()

def matplotlib_colors_LinearSegmentedColormap_set_extremes(self: 'any') -> 'any_computed':
    r"""
    Set the colors for masked (*bad*) values and, when ``norm.clip =
    False``, low (*under*) and high (*over*) out-of-range values.
    Keywords: matplotlib.colors.LinearSegmentedColormap, set_extremes
    """
    output_var = input_var.set_extremes()

def matplotlib_colors_LinearSegmentedColormap_set_gamma(self: 'any') -> 'any_computed':
    r"""
    Set a new gamma value and regenerate colormap.
    Keywords: matplotlib.colors.LinearSegmentedColormap, set_gamma
    """
    output_var = input_var.set_gamma()

def matplotlib_colors_LinearSegmentedColormap_set_over(self: 'any') -> 'any_computed':
    r"""
    Set the color for high out-of-range values.
    Keywords: matplotlib.colors.LinearSegmentedColormap, set_over
    """
    output_var = input_var.set_over()

def matplotlib_colors_LinearSegmentedColormap_set_under(self: 'any') -> 'any_computed':
    r"""
    Set the color for low out-of-range values.
    Keywords: matplotlib.colors.LinearSegmentedColormap, set_under
    """
    output_var = input_var.set_under()

def matplotlib_colors_LinearSegmentedColormap_with_extremes(self: 'any') -> 'any_computed':
    r"""
    Return a copy of the colormap, for which the colors for masked (*bad*)
    values and, when ``norm.clip = False``, low (*under*) and high (*over*)
    out-of-range values, have been set accordingly.
    Keywords: matplotlib.colors.LinearSegmentedColormap, with_extremes
    """
    output_var = input_var.with_extremes()

def matplotlib_colors_ListedColormap_copy(self: 'any') -> 'any_computed':
    r"""
    Return a copy of the colormap.
    Keywords: matplotlib.colors.ListedColormap, copy
    """
    output_var = input_var.copy()

def matplotlib_colors_ListedColormap_get_bad(self: 'any') -> 'any_computed':
    r"""
    Get the color for masked values.
    Keywords: matplotlib.colors.ListedColormap, get_bad
    """
    output_var = input_var.get_bad()

def matplotlib_colors_ListedColormap_get_over(self: 'any') -> 'any_computed':
    r"""
    Get the color for high out-of-range values.
    Keywords: matplotlib.colors.ListedColormap, get_over
    """
    output_var = input_var.get_over()

def matplotlib_colors_ListedColormap_get_under(self: 'any') -> 'any_computed':
    r"""
    Get the color for low out-of-range values.
    Keywords: matplotlib.colors.ListedColormap, get_under
    """
    output_var = input_var.get_under()

def matplotlib_colors_ListedColormap_is_gray(self: 'any') -> 'any_computed':
    r"""
    Return whether the colormap is grayscale.
    Keywords: matplotlib.colors.ListedColormap, is_gray
    """
    output_var = input_var.is_gray()

def matplotlib_colors_ListedColormap_resampled(self: 'any') -> 'any_computed':
    r"""
    Return a new colormap with *lutsize* entries.
    Keywords: matplotlib.colors.ListedColormap, resampled
    """
    output_var = input_var.resampled()

def matplotlib_colors_ListedColormap_reversed(self: 'any') -> 'any_computed':
    r"""
    Return a reversed instance of the Colormap.
    Parameters
    ----------
    name : str, optional
    Keywords: matplotlib.colors.ListedColormap, reversed
    """
    output_var = input_var.reversed()

def matplotlib_colors_ListedColormap_set_bad(self: 'any') -> 'any_computed':
    r"""
    Set the color for masked values.
    Keywords: matplotlib.colors.ListedColormap, set_bad
    """
    output_var = input_var.set_bad()

def matplotlib_colors_ListedColormap_set_extremes(self: 'any') -> 'any_computed':
    r"""
    Set the colors for masked (*bad*) values and, when ``norm.clip =
    False``, low (*under*) and high (*over*) out-of-range values.
    Keywords: matplotlib.colors.ListedColormap, set_extremes
    """
    output_var = input_var.set_extremes()

def matplotlib_colors_ListedColormap_set_over(self: 'any') -> 'any_computed':
    r"""
    Set the color for high out-of-range values.
    Keywords: matplotlib.colors.ListedColormap, set_over
    """
    output_var = input_var.set_over()

def matplotlib_colors_ListedColormap_set_under(self: 'any') -> 'any_computed':
    r"""
    Set the color for low out-of-range values.
    Keywords: matplotlib.colors.ListedColormap, set_under
    """
    output_var = input_var.set_under()

def matplotlib_colors_ListedColormap_with_extremes(self: 'any') -> 'any_computed':
    r"""
    Return a copy of the colormap, for which the colors for masked (*bad*)
    values and, when ``norm.clip = False``, low (*under*) and high (*over*)
    out-of-range values, have been set accordingly.
    Keywords: matplotlib.colors.ListedColormap, with_extremes
    """
    output_var = input_var.with_extremes()

def matplotlib_colors_LogNorm_autoscale(self: 'any') -> 'any_computed':
    r"""
    Set *vmin*, *vmax* to min, max of *A*.
    Keywords: matplotlib.colors.LogNorm, autoscale
    """
    output_var = input_var.autoscale()

def matplotlib_colors_LogNorm_autoscale_None(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.colors.LogNorm, autoscale_None
    """
    output_var = input_var.autoscale_None()

def matplotlib_colors_LogNorm_inverse(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.colors.LogNorm, inverse
    """
    output_var = input_var.inverse()

def matplotlib_colors_LogNorm_process_value(value: 'any') -> 'any_computed':
    r"""
    Homogenize the input *value* for easy and efficient normalization.
    *value* can be a scalar or sequence.
    Parameters
    Keywords: matplotlib.colors.LogNorm, process_value
    """
    output_var = input_var.process_value()

def matplotlib_colors_LogNorm_scaled(self: 'any') -> 'any_computed':
    r"""
    Return whether *vmin* and *vmax* are both set.
    Keywords: matplotlib.colors.LogNorm, scaled
    """
    output_var = input_var.scaled()

def matplotlib_colors_MultivarColormap_copy(self: 'any') -> 'any_computed':
    r"""
    Return a copy of the multivarcolormap.
    Keywords: matplotlib.colors.MultivarColormap, copy
    """
    output_var = input_var.copy()

def matplotlib_colors_MultivarColormap_get_bad(self: 'any') -> 'any_computed':
    r"""
    Get the color for masked values.
    Keywords: matplotlib.colors.MultivarColormap, get_bad
    """
    output_var = input_var.get_bad()

def matplotlib_colors_MultivarColormap_resampled(self: 'any') -> 'any_computed':
    r"""
    Return a new colormap with *lutshape* entries.
    Parameters
    ----------
    lutshape : tuple of (`int`, `None`)
    Keywords: matplotlib.colors.MultivarColormap, resampled
    """
    output_var = input_var.resampled()

def matplotlib_colors_MultivarColormap_with_extremes(self: 'any') -> 'any_computed':
    r"""
    Return a copy of the `MultivarColormap` with modified out-of-range attributes.
    The *bad* keyword modifies the copied `MultivarColormap` while *under* and
    *over* modifies the attributes of the copied component colormaps.
    Note that *under* and *over* colors are subject to the mixing rules determined
    Keywords: matplotlib.colors.MultivarColormap, with_extremes
    """
    output_var = input_var.with_extremes()

def matplotlib_colors_NoNorm_autoscale(self: 'any') -> 'any_computed':
    r"""
    Set *vmin*, *vmax* to min, max of *A*.
    Keywords: matplotlib.colors.NoNorm, autoscale
    """
    output_var = input_var.autoscale()

def matplotlib_colors_NoNorm_autoscale_None(self: 'any') -> 'any_computed':
    r"""
    If *vmin* or *vmax* are not set, use the min/max of *A* to set them.
    Keywords: matplotlib.colors.NoNorm, autoscale_None
    """
    output_var = input_var.autoscale_None()

def matplotlib_colors_NoNorm_inverse(self: 'any') -> 'any_computed':
    r"""
    Maps the normalized value (i.e., index in the colormap) back to image
    data value.
    Parameters
    ----------
    Keywords: matplotlib.colors.NoNorm, inverse
    """
    output_var = input_var.inverse()

def matplotlib_colors_NoNorm_process_value(value: 'any') -> 'any_computed':
    r"""
    Homogenize the input *value* for easy and efficient normalization.
    *value* can be a scalar or sequence.
    Parameters
    Keywords: matplotlib.colors.NoNorm, process_value
    """
    output_var = input_var.process_value()

def matplotlib_colors_NoNorm_scaled(self: 'any') -> 'any_computed':
    r"""
    Return whether *vmin* and *vmax* are both set.
    Keywords: matplotlib.colors.NoNorm, scaled
    """
    output_var = input_var.scaled()

def matplotlib_colors_Normalize_autoscale(self: 'any') -> 'any_computed':
    r"""
    Set *vmin*, *vmax* to min, max of *A*.
    Keywords: matplotlib.colors.Normalize, autoscale
    """
    output_var = input_var.autoscale()

def matplotlib_colors_Normalize_autoscale_None(self: 'any') -> 'any_computed':
    r"""
    If *vmin* or *vmax* are not set, use the min/max of *A* to set them.
    Keywords: matplotlib.colors.Normalize, autoscale_None
    """
    output_var = input_var.autoscale_None()

def matplotlib_colors_Normalize_inverse(self: 'any') -> 'any_computed':
    r"""
    Maps the normalized value (i.e., index in the colormap) back to image
    data value.
    Parameters
    ----------
    Keywords: matplotlib.colors.Normalize, inverse
    """
    output_var = input_var.inverse()

def matplotlib_colors_Normalize_process_value(value: 'any') -> 'any_computed':
    r"""
    Homogenize the input *value* for easy and efficient normalization.
    *value* can be a scalar or sequence.
    Parameters
    Keywords: matplotlib.colors.Normalize, process_value
    """
    output_var = input_var.process_value()

def matplotlib_colors_Normalize_scaled(self: 'any') -> 'any_computed':
    r"""
    Return whether *vmin* and *vmax* are both set.
    Keywords: matplotlib.colors.Normalize, scaled
    """
    output_var = input_var.scaled()

def matplotlib_colors_PowerNorm_autoscale(self: 'any') -> 'any_computed':
    r"""
    Set *vmin*, *vmax* to min, max of *A*.
    Keywords: matplotlib.colors.PowerNorm, autoscale
    """
    output_var = input_var.autoscale()

def matplotlib_colors_PowerNorm_autoscale_None(self: 'any') -> 'any_computed':
    r"""
    If *vmin* or *vmax* are not set, use the min/max of *A* to set them.
    Keywords: matplotlib.colors.PowerNorm, autoscale_None
    """
    output_var = input_var.autoscale_None()

def matplotlib_colors_PowerNorm_inverse(self: 'any') -> 'any_computed':
    r"""
    Maps the normalized value (i.e., index in the colormap) back to image
    data value.
    Parameters
    ----------
    Keywords: matplotlib.colors.PowerNorm, inverse
    """
    output_var = input_var.inverse()

def matplotlib_colors_PowerNorm_process_value(value: 'any') -> 'any_computed':
    r"""
    Homogenize the input *value* for easy and efficient normalization.
    *value* can be a scalar or sequence.
    Parameters
    Keywords: matplotlib.colors.PowerNorm, process_value
    """
    output_var = input_var.process_value()

def matplotlib_colors_PowerNorm_scaled(self: 'any') -> 'any_computed':
    r"""
    Return whether *vmin* and *vmax* are both set.
    Keywords: matplotlib.colors.PowerNorm, scaled
    """
    output_var = input_var.scaled()

def matplotlib_colors_SegmentedBivarColormap_copy(self: 'any') -> 'any_computed':
    r"""
    Return a copy of the colormap.
    Keywords: matplotlib.colors.SegmentedBivarColormap, copy
    """
    output_var = input_var.copy()

def matplotlib_colors_SegmentedBivarColormap_get_bad(self: 'any') -> 'any_computed':
    r"""
    Get the color for masked values.
    Keywords: matplotlib.colors.SegmentedBivarColormap, get_bad
    """
    output_var = input_var.get_bad()

def matplotlib_colors_SegmentedBivarColormap_get_outside(self: 'any') -> 'any_computed':
    r"""
    Get the color for out-of-range values.
    Keywords: matplotlib.colors.SegmentedBivarColormap, get_outside
    """
    output_var = input_var.get_outside()

def matplotlib_colors_SegmentedBivarColormap_resampled(self: 'any') -> 'any_computed':
    r"""
    Return a new colormap with *lutshape* entries.
    Note that this function does not move the origin.
    Parameters
    Keywords: matplotlib.colors.SegmentedBivarColormap, resampled
    """
    output_var = input_var.resampled()

def matplotlib_colors_SegmentedBivarColormap_reversed(self: 'any') -> 'any_computed':
    r"""
    Reverses both or one of the axis.
    Keywords: matplotlib.colors.SegmentedBivarColormap, reversed
    """
    output_var = input_var.reversed()

def matplotlib_colors_SegmentedBivarColormap_transposed(self: 'any') -> 'any_computed':
    r"""
    Transposes the colormap by swapping the order of the axis
    Keywords: matplotlib.colors.SegmentedBivarColormap, transposed
    """
    output_var = input_var.transposed()

def matplotlib_colors_SegmentedBivarColormap_with_extremes(self: 'any') -> 'any_computed':
    r"""
    Return a copy of the `BivarColormap` with modified attributes.
    Note that the *outside* color is only relevant if `shape` = 'ignore'
    or 'circleignore'.
    Keywords: matplotlib.colors.SegmentedBivarColormap, with_extremes
    """
    output_var = input_var.with_extremes()

def matplotlib_colors_SymLogNorm_autoscale(self: 'any') -> 'any_computed':
    r"""
    Set *vmin*, *vmax* to min, max of *A*.
    Keywords: matplotlib.colors.SymLogNorm, autoscale
    """
    output_var = input_var.autoscale()

def matplotlib_colors_SymLogNorm_autoscale_None(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.colors.SymLogNorm, autoscale_None
    """
    output_var = input_var.autoscale_None()

def matplotlib_colors_SymLogNorm_inverse(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.colors.SymLogNorm, inverse
    """
    output_var = input_var.inverse()

def matplotlib_colors_SymLogNorm_process_value(value: 'any') -> 'any_computed':
    r"""
    Homogenize the input *value* for easy and efficient normalization.
    *value* can be a scalar or sequence.
    Parameters
    Keywords: matplotlib.colors.SymLogNorm, process_value
    """
    output_var = input_var.process_value()

def matplotlib_colors_SymLogNorm_scaled(self: 'any') -> 'any_computed':
    r"""
    Return whether *vmin* and *vmax* are both set.
    Keywords: matplotlib.colors.SymLogNorm, scaled
    """
    output_var = input_var.scaled()

def matplotlib_colors_TwoSlopeNorm_autoscale(self: 'any') -> 'any_computed':
    r"""
    Set *vmin*, *vmax* to min, max of *A*.
    Keywords: matplotlib.colors.TwoSlopeNorm, autoscale
    """
    output_var = input_var.autoscale()

def matplotlib_colors_TwoSlopeNorm_autoscale_None(self: 'any') -> 'any_computed':
    r"""
    Get vmin and vmax.
    If vcenter isn't in the range [vmin, vmax], either vmin or vmax
    is expanded so that vcenter lies in the middle of the modified range
    [vmin, vmax].
    Keywords: matplotlib.colors.TwoSlopeNorm, autoscale_None
    """
    output_var = input_var.autoscale_None()

def matplotlib_colors_TwoSlopeNorm_inverse(self: 'any') -> 'any_computed':
    r"""
    Maps the normalized value (i.e., index in the colormap) back to image
    data value.
    Parameters
    ----------
    Keywords: matplotlib.colors.TwoSlopeNorm, inverse
    """
    output_var = input_var.inverse()

def matplotlib_colors_TwoSlopeNorm_process_value(value: 'any') -> 'any_computed':
    r"""
    Homogenize the input *value* for easy and efficient normalization.
    *value* can be a scalar or sequence.
    Parameters
    Keywords: matplotlib.colors.TwoSlopeNorm, process_value
    """
    output_var = input_var.process_value()

def matplotlib_colors_TwoSlopeNorm_scaled(self: 'any') -> 'any_computed':
    r"""
    Return whether *vmin* and *vmax* are both set.
    Keywords: matplotlib.colors.TwoSlopeNorm, scaled
    """
    output_var = input_var.scaled()

def matplotlib_colors_from_levels_and_colors(levels: 'any') -> 'any_computed':
    r"""
    A helper routine to generate a cmap and a norm instance which
    behave similar to contourf's levels and colors arguments.
    Parameters
    ----------
    Keywords: matplotlib.colors, from_levels_and_colors
    """
    output_var = matplotlib.colors.from_levels_and_colors(input_var)

def matplotlib_colors_hex2color(c: 'any') -> 'any_computed':
    r"""
    Convert *c* to an RGB color, silently dropping the alpha channel.
    Keywords: matplotlib.colors, hex2color
    """
    output_var = matplotlib.colors.hex2color(input_var)

def matplotlib_colors_hsv_to_rgb(hsv: 'any') -> 'any_computed':
    r"""
    Convert HSV values to RGB.
    Parameters
    ----------
    hsv : (..., 3) array-like
    Keywords: matplotlib.colors, hsv_to_rgb
    """
    output_var = matplotlib.colors.hsv_to_rgb(input_var)

def matplotlib_colors_is_color_like(c: 'any') -> 'any_computed':
    r"""
    Return whether *c* can be interpreted as an RGB(A) color.
    Keywords: matplotlib.colors, is_color_like
    """
    output_var = matplotlib.colors.is_color_like(input_var)

def matplotlib_colors_make_norm_from_scale(scale_cls: 'any') -> 'any_computed':
    r"""
    Decorator for building a `.Normalize` subclass from a `~.scale.ScaleBase`
    subclass.
    After ::
    Keywords: matplotlib.colors, make_norm_from_scale
    """
    output_var = matplotlib.colors.make_norm_from_scale(input_var)

def matplotlib_colors_rgb2hex(c: 'any') -> 'any_computed':
    r"""
    Convert *c* to a hex color.
    Parameters
    ----------
    c : :ref:`color <colors_def>` or `numpy.ma.masked`
    Keywords: matplotlib.colors, rgb2hex
    """
    output_var = matplotlib.colors.rgb2hex(input_var)

def matplotlib_colors_rgb_to_hsv(arr: 'any') -> 'any_computed':
    r"""
    Convert an array of float RGB values (in the range [0, 1]) to HSV values.
    Parameters
    ----------
    arr : (..., 3) array-like
    Keywords: matplotlib.colors, rgb_to_hsv
    """
    output_var = matplotlib.colors.rgb_to_hsv(input_var)

def matplotlib_colors_same_color(c1: 'any') -> 'any_computed':
    r"""
    Return whether the colors *c1* and *c2* are the same.
    *c1*, *c2* can be single colors or lists/arrays of colors.
    Keywords: matplotlib.colors, same_color
    """
    output_var = matplotlib.colors.same_color(input_var)

def matplotlib_scale_AsinhLocator_create_dummy_axis(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.scale.AsinhLocator, create_dummy_axis
    """
    output_var = input_var.create_dummy_axis()

def matplotlib_scale_AsinhLocator_nonsingular(self: 'any') -> 'any_computed':
    r"""
    Adjust a range as needed to avoid singularities.
    This method gets called during autoscaling, with ``(v0, v1)`` set to
    the data limits on the Axes if the Axes contains any data, or
    ``(-inf, +inf)`` if not.
    Keywords: matplotlib.scale.AsinhLocator, nonsingular
    """
    output_var = input_var.nonsingular()

def matplotlib_scale_AsinhLocator_raise_if_exceeds(self: 'any') -> 'any_computed':
    r"""
    Log at WARNING level if *locs* is longer than `Locator.MAXTICKS`.
    This is intended to be called immediately before returning *locs* from
    ``__call__`` to inform users in case their Locator returns a huge
    number of ticks, causing Matplotlib to run out of memory.
    Keywords: matplotlib.scale.AsinhLocator, raise_if_exceeds
    """
    output_var = input_var.raise_if_exceeds()

def matplotlib_scale_AsinhLocator_set_axis(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.scale.AsinhLocator, set_axis
    """
    output_var = input_var.set_axis()

def matplotlib_scale_AsinhLocator_set_params(self: 'any') -> 'any_computed':
    r"""
    Set parameters within this locator.
    Keywords: matplotlib.scale.AsinhLocator, set_params
    """
    output_var = input_var.set_params()

def matplotlib_scale_AsinhLocator_tick_values(self: 'any') -> 'any_computed':
    r"""
    Return the values of the located ticks given **vmin** and **vmax**.
    .. note::
        To get tick locations with the vmin and vmax values defined
        automatically for the associated ``axis`` simply call
    Keywords: matplotlib.scale.AsinhLocator, tick_values
    """
    output_var = input_var.tick_values()

def matplotlib_scale_AsinhLocator_view_limits(self: 'any') -> 'any_computed':
    r"""
    Select a scale for the range from vmin to vmax.
    Subclasses should override this method to change locator behaviour.
    Keywords: matplotlib.scale.AsinhLocator, view_limits
    """
    output_var = input_var.view_limits()

def matplotlib_scale_AsinhScale_get_transform(self: 'any') -> 'any_computed':
    r"""
    Return the `.Transform` object associated with this scale.
    Keywords: matplotlib.scale.AsinhScale, get_transform
    """
    output_var = input_var.get_transform()

def matplotlib_scale_AsinhScale_limit_range_for_scale(self: 'any') -> 'any_computed':
    r"""
    Return the range *vmin*, *vmax*, restricted to the
    domain supported by this scale (if any).
    *minpos* should be the minimum positive value in the data.
    This is used by log scales to determine a minimum value.
    Keywords: matplotlib.scale.AsinhScale, limit_range_for_scale
    """
    output_var = input_var.limit_range_for_scale()

def matplotlib_scale_AsinhScale_set_default_locators_and_formatters(self: 'any') -> 'any_computed':
    r"""
    Set the locators and formatters of *axis* to instances suitable for
    this scale.
    Keywords: matplotlib.scale.AsinhScale, set_default_locators_and_formatters
    """
    output_var = input_var.set_default_locators_and_formatters()

def matplotlib_scale_AsinhTransform_contains_branch(self: 'any') -> 'any_computed':
    r"""
    Return whether the given transform is a sub-tree of this transform.
    This routine uses transform equality to identify sub-trees, therefore
    in many situations it is object id which will be used.
    Keywords: matplotlib.scale.AsinhTransform, contains_branch
    """
    output_var = input_var.contains_branch()

def matplotlib_scale_AsinhTransform_contains_branch_seperately(self: 'any') -> 'any_computed':
    r"""
    Return whether the given branch is a sub-tree of this transform on
    each separate dimension.
    A common use for this method is to identify if a transform is a blended
    transform containing an Axes' data transform. e.g.::
    Keywords: matplotlib.scale.AsinhTransform, contains_branch_seperately
    """
    output_var = input_var.contains_branch_seperately()

def matplotlib_scale_AsinhTransform_frozen(self: 'any') -> 'any_computed':
    r"""
    Return a frozen copy of this transform node.  The frozen copy will not
    be updated when its children change.  Useful for storing a previously
    known state of a transform where ``copy.deepcopy()`` might normally be
    used.
    Keywords: matplotlib.scale.AsinhTransform, frozen
    """
    output_var = input_var.frozen()

def matplotlib_scale_AsinhTransform_get_affine(self: 'any') -> 'any_computed':
    r"""
    Get the affine part of this transform.
    Keywords: matplotlib.scale.AsinhTransform, get_affine
    """
    output_var = input_var.get_affine()

def matplotlib_scale_AsinhTransform_get_matrix(self: 'any') -> 'any_computed':
    r"""
    Get the matrix for the affine part of this transform.
    Keywords: matplotlib.scale.AsinhTransform, get_matrix
    """
    output_var = input_var.get_matrix()

def matplotlib_scale_AsinhTransform_invalidate(self: 'any') -> 'any_computed':
    r"""
    Invalidate this `TransformNode` and triggers an invalidation of its
    ancestors.  Should be called any time the transform changes.
    Keywords: matplotlib.scale.AsinhTransform, invalidate
    """
    output_var = input_var.invalidate()

def matplotlib_scale_AsinhTransform_inverted(self: 'any') -> 'any_computed':
    r"""
    Return the corresponding inverse transformation.
    It holds ``x == self.inverted().transform(self.transform(x))``.
    The return value of this method should be treated as
    Keywords: matplotlib.scale.AsinhTransform, inverted
    """
    output_var = input_var.inverted()

def matplotlib_scale_AsinhTransform_set_children(self: 'any') -> 'any_computed':
    r"""
    Set the children of the transform, to let the invalidation
    system know which transforms can invalidate this transform.
    Should be called from the constructor of any transforms that
    depend on other transforms.
    Keywords: matplotlib.scale.AsinhTransform, set_children
    """
    output_var = input_var.set_children()

def matplotlib_scale_AsinhTransform_transform(self: 'any') -> 'any_computed':
    r"""
    Apply this transformation on the given array of *values*.
    Parameters
    ----------
    values : array-like
    Keywords: matplotlib.scale.AsinhTransform, transform
    """
    output_var = input_var.transform()

def matplotlib_scale_AsinhTransform_transform_affine(self: 'any') -> 'any_computed':
    r"""
    Apply only the affine part of this transformation on the
    given array of values.
    ``transform(values)`` is always equivalent to
    ``transform_affine(transform_non_affine(values))``.
    Keywords: matplotlib.scale.AsinhTransform, transform_affine
    """
    output_var = input_var.transform_affine()

def matplotlib_scale_AsinhTransform_transform_angles(self: 'any') -> 'any_computed':
    r"""
    Transform a set of angles anchored at specific locations.
    Parameters
    ----------
    angles : (N,) array-like
    Keywords: matplotlib.scale.AsinhTransform, transform_angles
    """
    output_var = input_var.transform_angles()

def matplotlib_scale_AsinhTransform_transform_bbox(self: 'any') -> 'any_computed':
    r"""
    Transform the given bounding box.
    For smarter transforms including caching (a common requirement in
    Matplotlib), see `TransformedBbox`.
    Keywords: matplotlib.scale.AsinhTransform, transform_bbox
    """
    output_var = input_var.transform_bbox()

def matplotlib_scale_AsinhTransform_transform_non_affine(self: 'any') -> 'any_computed':
    r"""
    Apply only the non-affine part of this transformation.
    ``transform(values)`` is always equivalent to
    ``transform_affine(transform_non_affine(values))``.
    Keywords: matplotlib.scale.AsinhTransform, transform_non_affine
    """
    output_var = input_var.transform_non_affine()

def matplotlib_scale_AsinhTransform_transform_path(self: 'any') -> 'any_computed':
    r"""
    Apply the transform to `.Path` *path*, returning a new `.Path`.
    In some cases, this transform may insert curves into the path
    that began as line segments.
    Keywords: matplotlib.scale.AsinhTransform, transform_path
    """
    output_var = input_var.transform_path()

def matplotlib_scale_AsinhTransform_transform_path_affine(self: 'any') -> 'any_computed':
    r"""
    Apply the affine part of this transform to `.Path` *path*, returning a
    new `.Path`.
    ``transform_path(path)`` is equivalent to
    ``transform_path_affine(transform_path_non_affine(values))``.
    Keywords: matplotlib.scale.AsinhTransform, transform_path_affine
    """
    output_var = input_var.transform_path_affine()

def matplotlib_scale_AsinhTransform_transform_path_non_affine(self: 'any') -> 'any_computed':
    r"""
    Apply the non-affine part of this transform to `.Path` *path*,
    returning a new `.Path`.
    ``transform_path(path)`` is equivalent to
    ``transform_path_affine(transform_path_non_affine(values))``.
    Keywords: matplotlib.scale.AsinhTransform, transform_path_non_affine
    """
    output_var = input_var.transform_path_non_affine()

def matplotlib_scale_AsinhTransform_transform_point(self: 'any') -> 'any_computed':
    r"""
    Return a transformed point.
    This function is only kept for backcompatibility; the more general
    `.transform` method is capable of transforming both a list of points
    and a single point.
    Keywords: matplotlib.scale.AsinhTransform, transform_point
    """
    output_var = input_var.transform_point()

def matplotlib_scale_AutoLocator_create_dummy_axis(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.scale.AutoLocator, create_dummy_axis
    """
    output_var = input_var.create_dummy_axis()

def matplotlib_scale_AutoLocator_nonsingular(self: 'any') -> 'any_computed':
    r"""
    Adjust a range as needed to avoid singularities.
    This method gets called during autoscaling, with ``(v0, v1)`` set to
    the data limits on the Axes if the Axes contains any data, or
    ``(-inf, +inf)`` if not.
    Keywords: matplotlib.scale.AutoLocator, nonsingular
    """
    output_var = input_var.nonsingular()

def matplotlib_scale_AutoLocator_raise_if_exceeds(self: 'any') -> 'any_computed':
    r"""
    Log at WARNING level if *locs* is longer than `Locator.MAXTICKS`.
    This is intended to be called immediately before returning *locs* from
    ``__call__`` to inform users in case their Locator returns a huge
    number of ticks, causing Matplotlib to run out of memory.
    Keywords: matplotlib.scale.AutoLocator, raise_if_exceeds
    """
    output_var = input_var.raise_if_exceeds()

def matplotlib_scale_AutoLocator_set_axis(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.scale.AutoLocator, set_axis
    """
    output_var = input_var.set_axis()

def matplotlib_scale_AutoLocator_set_params(self: 'any') -> 'any_computed':
    r"""
    Set parameters for this locator.
    Parameters
    ----------
    nbins : int or 'auto', optional
    Keywords: matplotlib.scale.AutoLocator, set_params
    """
    output_var = input_var.set_params()

def matplotlib_scale_AutoLocator_tick_values(self: 'any') -> 'any_computed':
    r"""
    Return the values of the located ticks given **vmin** and **vmax**.
    .. note::
        To get tick locations with the vmin and vmax values defined
        automatically for the associated ``axis`` simply call
    Keywords: matplotlib.scale.AutoLocator, tick_values
    """
    output_var = input_var.tick_values()

def matplotlib_scale_AutoLocator_view_limits(self: 'any') -> 'any_computed':
    r"""
    Select a scale for the range from vmin to vmax.
    Subclasses should override this method to change locator behaviour.
    Keywords: matplotlib.scale.AutoLocator, view_limits
    """
    output_var = input_var.view_limits()

def matplotlib_scale_AutoMinorLocator_create_dummy_axis(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.scale.AutoMinorLocator, create_dummy_axis
    """
    output_var = input_var.create_dummy_axis()

def matplotlib_scale_AutoMinorLocator_nonsingular(self: 'any') -> 'any_computed':
    r"""
    Adjust a range as needed to avoid singularities.
    This method gets called during autoscaling, with ``(v0, v1)`` set to
    the data limits on the Axes if the Axes contains any data, or
    ``(-inf, +inf)`` if not.
    Keywords: matplotlib.scale.AutoMinorLocator, nonsingular
    """
    output_var = input_var.nonsingular()

def matplotlib_scale_AutoMinorLocator_raise_if_exceeds(self: 'any') -> 'any_computed':
    r"""
    Log at WARNING level if *locs* is longer than `Locator.MAXTICKS`.
    This is intended to be called immediately before returning *locs* from
    ``__call__`` to inform users in case their Locator returns a huge
    number of ticks, causing Matplotlib to run out of memory.
    Keywords: matplotlib.scale.AutoMinorLocator, raise_if_exceeds
    """
    output_var = input_var.raise_if_exceeds()

def matplotlib_scale_AutoMinorLocator_set_axis(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.scale.AutoMinorLocator, set_axis
    """
    output_var = input_var.set_axis()

def matplotlib_scale_AutoMinorLocator_set_params(self: 'any') -> 'any_computed':
    r"""
    Do nothing, and raise a warning. Any locator class not supporting the
    set_params() function will call this.
    Keywords: matplotlib.scale.AutoMinorLocator, set_params
    """
    output_var = input_var.set_params()

def matplotlib_scale_AutoMinorLocator_tick_values(self: 'any') -> 'any_computed':
    r"""
    Return the values of the located ticks given **vmin** and **vmax**.
    .. note::
        To get tick locations with the vmin and vmax values defined
        automatically for the associated ``axis`` simply call
    Keywords: matplotlib.scale.AutoMinorLocator, tick_values
    """
    output_var = input_var.tick_values()

def matplotlib_scale_AutoMinorLocator_view_limits(self: 'any') -> 'any_computed':
    r"""
    Select a scale for the range from vmin to vmax.
    Subclasses should override this method to change locator behaviour.
    Keywords: matplotlib.scale.AutoMinorLocator, view_limits
    """
    output_var = input_var.view_limits()

def matplotlib_scale_FuncScale_get_transform(self: 'any') -> 'any_computed':
    r"""
    Return the `.FuncTransform` associated with this scale.
    Keywords: matplotlib.scale.FuncScale, get_transform
    """
    output_var = input_var.get_transform()

def matplotlib_scale_FuncScale_limit_range_for_scale(self: 'any') -> 'any_computed':
    r"""
    Return the range *vmin*, *vmax*, restricted to the
    domain supported by this scale (if any).
    *minpos* should be the minimum positive value in the data.
    This is used by log scales to determine a minimum value.
    Keywords: matplotlib.scale.FuncScale, limit_range_for_scale
    """
    output_var = input_var.limit_range_for_scale()

def matplotlib_scale_FuncScale_set_default_locators_and_formatters(self: 'any') -> 'any_computed':
    r"""
    Set the locators and formatters of *axis* to instances suitable for
    this scale.
    Keywords: matplotlib.scale.FuncScale, set_default_locators_and_formatters
    """
    output_var = input_var.set_default_locators_and_formatters()

def matplotlib_scale_FuncScaleLog_get_transform(self: 'any') -> 'any_computed':
    r"""
    Return the `.Transform` associated with this scale.
    Keywords: matplotlib.scale.FuncScaleLog, get_transform
    """
    output_var = input_var.get_transform()

def matplotlib_scale_FuncScaleLog_limit_range_for_scale(self: 'any') -> 'any_computed':
    r"""
    Limit the domain to positive values.
    Keywords: matplotlib.scale.FuncScaleLog, limit_range_for_scale
    """
    output_var = input_var.limit_range_for_scale()

def matplotlib_scale_FuncScaleLog_set_default_locators_and_formatters(self: 'any') -> 'any_computed':
    r"""
    Set the locators and formatters of *axis* to instances suitable for
    this scale.
    Keywords: matplotlib.scale.FuncScaleLog, set_default_locators_and_formatters
    """
    output_var = input_var.set_default_locators_and_formatters()

def matplotlib_scale_FuncTransform_contains_branch(self: 'any') -> 'any_computed':
    r"""
    Return whether the given transform is a sub-tree of this transform.
    This routine uses transform equality to identify sub-trees, therefore
    in many situations it is object id which will be used.
    Keywords: matplotlib.scale.FuncTransform, contains_branch
    """
    output_var = input_var.contains_branch()

def matplotlib_scale_FuncTransform_contains_branch_seperately(self: 'any') -> 'any_computed':
    r"""
    Return whether the given branch is a sub-tree of this transform on
    each separate dimension.
    A common use for this method is to identify if a transform is a blended
    transform containing an Axes' data transform. e.g.::
    Keywords: matplotlib.scale.FuncTransform, contains_branch_seperately
    """
    output_var = input_var.contains_branch_seperately()

def matplotlib_scale_FuncTransform_frozen(self: 'any') -> 'any_computed':
    r"""
    Return a frozen copy of this transform node.  The frozen copy will not
    be updated when its children change.  Useful for storing a previously
    known state of a transform where ``copy.deepcopy()`` might normally be
    used.
    Keywords: matplotlib.scale.FuncTransform, frozen
    """
    output_var = input_var.frozen()

def matplotlib_scale_FuncTransform_get_affine(self: 'any') -> 'any_computed':
    r"""
    Get the affine part of this transform.
    Keywords: matplotlib.scale.FuncTransform, get_affine
    """
    output_var = input_var.get_affine()

def matplotlib_scale_FuncTransform_get_matrix(self: 'any') -> 'any_computed':
    r"""
    Get the matrix for the affine part of this transform.
    Keywords: matplotlib.scale.FuncTransform, get_matrix
    """
    output_var = input_var.get_matrix()

def matplotlib_scale_FuncTransform_invalidate(self: 'any') -> 'any_computed':
    r"""
    Invalidate this `TransformNode` and triggers an invalidation of its
    ancestors.  Should be called any time the transform changes.
    Keywords: matplotlib.scale.FuncTransform, invalidate
    """
    output_var = input_var.invalidate()

def matplotlib_scale_FuncTransform_inverted(self: 'any') -> 'any_computed':
    r"""
    Return the corresponding inverse transformation.
    It holds ``x == self.inverted().transform(self.transform(x))``.
    The return value of this method should be treated as
    Keywords: matplotlib.scale.FuncTransform, inverted
    """
    output_var = input_var.inverted()

def matplotlib_scale_FuncTransform_set_children(self: 'any') -> 'any_computed':
    r"""
    Set the children of the transform, to let the invalidation
    system know which transforms can invalidate this transform.
    Should be called from the constructor of any transforms that
    depend on other transforms.
    Keywords: matplotlib.scale.FuncTransform, set_children
    """
    output_var = input_var.set_children()

def matplotlib_scale_FuncTransform_transform(self: 'any') -> 'any_computed':
    r"""
    Apply this transformation on the given array of *values*.
    Parameters
    ----------
    values : array-like
    Keywords: matplotlib.scale.FuncTransform, transform
    """
    output_var = input_var.transform()

def matplotlib_scale_FuncTransform_transform_affine(self: 'any') -> 'any_computed':
    r"""
    Apply only the affine part of this transformation on the
    given array of values.
    ``transform(values)`` is always equivalent to
    ``transform_affine(transform_non_affine(values))``.
    Keywords: matplotlib.scale.FuncTransform, transform_affine
    """
    output_var = input_var.transform_affine()

def matplotlib_scale_FuncTransform_transform_angles(self: 'any') -> 'any_computed':
    r"""
    Transform a set of angles anchored at specific locations.
    Parameters
    ----------
    angles : (N,) array-like
    Keywords: matplotlib.scale.FuncTransform, transform_angles
    """
    output_var = input_var.transform_angles()

def matplotlib_scale_FuncTransform_transform_bbox(self: 'any') -> 'any_computed':
    r"""
    Transform the given bounding box.
    For smarter transforms including caching (a common requirement in
    Matplotlib), see `TransformedBbox`.
    Keywords: matplotlib.scale.FuncTransform, transform_bbox
    """
    output_var = input_var.transform_bbox()

def matplotlib_scale_FuncTransform_transform_non_affine(self: 'any') -> 'any_computed':
    r"""
    Apply only the non-affine part of this transformation.
    ``transform(values)`` is always equivalent to
    ``transform_affine(transform_non_affine(values))``.
    Keywords: matplotlib.scale.FuncTransform, transform_non_affine
    """
    output_var = input_var.transform_non_affine()

def matplotlib_scale_FuncTransform_transform_path(self: 'any') -> 'any_computed':
    r"""
    Apply the transform to `.Path` *path*, returning a new `.Path`.
    In some cases, this transform may insert curves into the path
    that began as line segments.
    Keywords: matplotlib.scale.FuncTransform, transform_path
    """
    output_var = input_var.transform_path()

def matplotlib_scale_FuncTransform_transform_path_affine(self: 'any') -> 'any_computed':
    r"""
    Apply the affine part of this transform to `.Path` *path*, returning a
    new `.Path`.
    ``transform_path(path)`` is equivalent to
    ``transform_path_affine(transform_path_non_affine(values))``.
    Keywords: matplotlib.scale.FuncTransform, transform_path_affine
    """
    output_var = input_var.transform_path_affine()

def matplotlib_scale_FuncTransform_transform_path_non_affine(self: 'any') -> 'any_computed':
    r"""
    Apply the non-affine part of this transform to `.Path` *path*,
    returning a new `.Path`.
    ``transform_path(path)`` is equivalent to
    ``transform_path_affine(transform_path_non_affine(values))``.
    Keywords: matplotlib.scale.FuncTransform, transform_path_non_affine
    """
    output_var = input_var.transform_path_non_affine()

def matplotlib_scale_FuncTransform_transform_point(self: 'any') -> 'any_computed':
    r"""
    Return a transformed point.
    This function is only kept for backcompatibility; the more general
    `.transform` method is capable of transforming both a list of points
    and a single point.
    Keywords: matplotlib.scale.FuncTransform, transform_point
    """
    output_var = input_var.transform_point()

def matplotlib_scale_InvertedAsinhTransform_contains_branch(self: 'any') -> 'any_computed':
    r"""
    Return whether the given transform is a sub-tree of this transform.
    This routine uses transform equality to identify sub-trees, therefore
    in many situations it is object id which will be used.
    Keywords: matplotlib.scale.InvertedAsinhTransform, contains_branch
    """
    output_var = input_var.contains_branch()

def matplotlib_scale_InvertedAsinhTransform_contains_branch_seperately(self: 'any') -> 'any_computed':
    r"""
    Return whether the given branch is a sub-tree of this transform on
    each separate dimension.
    A common use for this method is to identify if a transform is a blended
    transform containing an Axes' data transform. e.g.::
    Keywords: matplotlib.scale.InvertedAsinhTransform, contains_branch_seperately
    """
    output_var = input_var.contains_branch_seperately()

def matplotlib_scale_InvertedAsinhTransform_frozen(self: 'any') -> 'any_computed':
    r"""
    Return a frozen copy of this transform node.  The frozen copy will not
    be updated when its children change.  Useful for storing a previously
    known state of a transform where ``copy.deepcopy()`` might normally be
    used.
    Keywords: matplotlib.scale.InvertedAsinhTransform, frozen
    """
    output_var = input_var.frozen()

def matplotlib_scale_InvertedAsinhTransform_get_affine(self: 'any') -> 'any_computed':
    r"""
    Get the affine part of this transform.
    Keywords: matplotlib.scale.InvertedAsinhTransform, get_affine
    """
    output_var = input_var.get_affine()

def matplotlib_scale_InvertedAsinhTransform_get_matrix(self: 'any') -> 'any_computed':
    r"""
    Get the matrix for the affine part of this transform.
    Keywords: matplotlib.scale.InvertedAsinhTransform, get_matrix
    """
    output_var = input_var.get_matrix()

def matplotlib_scale_InvertedAsinhTransform_invalidate(self: 'any') -> 'any_computed':
    r"""
    Invalidate this `TransformNode` and triggers an invalidation of its
    ancestors.  Should be called any time the transform changes.
    Keywords: matplotlib.scale.InvertedAsinhTransform, invalidate
    """
    output_var = input_var.invalidate()

def matplotlib_scale_InvertedAsinhTransform_inverted(self: 'any') -> 'any_computed':
    r"""
    Return the corresponding inverse transformation.
    It holds ``x == self.inverted().transform(self.transform(x))``.
    The return value of this method should be treated as
    Keywords: matplotlib.scale.InvertedAsinhTransform, inverted
    """
    output_var = input_var.inverted()

def matplotlib_scale_InvertedAsinhTransform_set_children(self: 'any') -> 'any_computed':
    r"""
    Set the children of the transform, to let the invalidation
    system know which transforms can invalidate this transform.
    Should be called from the constructor of any transforms that
    depend on other transforms.
    Keywords: matplotlib.scale.InvertedAsinhTransform, set_children
    """
    output_var = input_var.set_children()

def matplotlib_scale_InvertedAsinhTransform_transform(self: 'any') -> 'any_computed':
    r"""
    Apply this transformation on the given array of *values*.
    Parameters
    ----------
    values : array-like
    Keywords: matplotlib.scale.InvertedAsinhTransform, transform
    """
    output_var = input_var.transform()

def matplotlib_scale_InvertedAsinhTransform_transform_affine(self: 'any') -> 'any_computed':
    r"""
    Apply only the affine part of this transformation on the
    given array of values.
    ``transform(values)`` is always equivalent to
    ``transform_affine(transform_non_affine(values))``.
    Keywords: matplotlib.scale.InvertedAsinhTransform, transform_affine
    """
    output_var = input_var.transform_affine()

def matplotlib_scale_InvertedAsinhTransform_transform_angles(self: 'any') -> 'any_computed':
    r"""
    Transform a set of angles anchored at specific locations.
    Parameters
    ----------
    angles : (N,) array-like
    Keywords: matplotlib.scale.InvertedAsinhTransform, transform_angles
    """
    output_var = input_var.transform_angles()

def matplotlib_scale_InvertedAsinhTransform_transform_bbox(self: 'any') -> 'any_computed':
    r"""
    Transform the given bounding box.
    For smarter transforms including caching (a common requirement in
    Matplotlib), see `TransformedBbox`.
    Keywords: matplotlib.scale.InvertedAsinhTransform, transform_bbox
    """
    output_var = input_var.transform_bbox()

def matplotlib_scale_InvertedAsinhTransform_transform_non_affine(self: 'any') -> 'any_computed':
    r"""
    Apply only the non-affine part of this transformation.
    ``transform(values)`` is always equivalent to
    ``transform_affine(transform_non_affine(values))``.
    Keywords: matplotlib.scale.InvertedAsinhTransform, transform_non_affine
    """
    output_var = input_var.transform_non_affine()

def matplotlib_scale_InvertedAsinhTransform_transform_path(self: 'any') -> 'any_computed':
    r"""
    Apply the transform to `.Path` *path*, returning a new `.Path`.
    In some cases, this transform may insert curves into the path
    that began as line segments.
    Keywords: matplotlib.scale.InvertedAsinhTransform, transform_path
    """
    output_var = input_var.transform_path()

def matplotlib_scale_InvertedAsinhTransform_transform_path_affine(self: 'any') -> 'any_computed':
    r"""
    Apply the affine part of this transform to `.Path` *path*, returning a
    new `.Path`.
    ``transform_path(path)`` is equivalent to
    ``transform_path_affine(transform_path_non_affine(values))``.
    Keywords: matplotlib.scale.InvertedAsinhTransform, transform_path_affine
    """
    output_var = input_var.transform_path_affine()

def matplotlib_scale_InvertedAsinhTransform_transform_path_non_affine(self: 'any') -> 'any_computed':
    r"""
    Apply the non-affine part of this transform to `.Path` *path*,
    returning a new `.Path`.
    ``transform_path(path)`` is equivalent to
    ``transform_path_affine(transform_path_non_affine(values))``.
    Keywords: matplotlib.scale.InvertedAsinhTransform, transform_path_non_affine
    """
    output_var = input_var.transform_path_non_affine()

def matplotlib_scale_InvertedAsinhTransform_transform_point(self: 'any') -> 'any_computed':
    r"""
    Return a transformed point.
    This function is only kept for backcompatibility; the more general
    `.transform` method is capable of transforming both a list of points
    and a single point.
    Keywords: matplotlib.scale.InvertedAsinhTransform, transform_point
    """
    output_var = input_var.transform_point()

def matplotlib_scale_InvertedLogTransform_contains_branch(self: 'any') -> 'any_computed':
    r"""
    Return whether the given transform is a sub-tree of this transform.
    This routine uses transform equality to identify sub-trees, therefore
    in many situations it is object id which will be used.
    Keywords: matplotlib.scale.InvertedLogTransform, contains_branch
    """
    output_var = input_var.contains_branch()

def matplotlib_scale_InvertedLogTransform_contains_branch_seperately(self: 'any') -> 'any_computed':
    r"""
    Return whether the given branch is a sub-tree of this transform on
    each separate dimension.
    A common use for this method is to identify if a transform is a blended
    transform containing an Axes' data transform. e.g.::
    Keywords: matplotlib.scale.InvertedLogTransform, contains_branch_seperately
    """
    output_var = input_var.contains_branch_seperately()

def matplotlib_scale_InvertedLogTransform_frozen(self: 'any') -> 'any_computed':
    r"""
    Return a frozen copy of this transform node.  The frozen copy will not
    be updated when its children change.  Useful for storing a previously
    known state of a transform where ``copy.deepcopy()`` might normally be
    used.
    Keywords: matplotlib.scale.InvertedLogTransform, frozen
    """
    output_var = input_var.frozen()

def matplotlib_scale_InvertedLogTransform_get_affine(self: 'any') -> 'any_computed':
    r"""
    Get the affine part of this transform.
    Keywords: matplotlib.scale.InvertedLogTransform, get_affine
    """
    output_var = input_var.get_affine()

def matplotlib_scale_InvertedLogTransform_get_matrix(self: 'any') -> 'any_computed':
    r"""
    Get the matrix for the affine part of this transform.
    Keywords: matplotlib.scale.InvertedLogTransform, get_matrix
    """
    output_var = input_var.get_matrix()

def matplotlib_scale_InvertedLogTransform_invalidate(self: 'any') -> 'any_computed':
    r"""
    Invalidate this `TransformNode` and triggers an invalidation of its
    ancestors.  Should be called any time the transform changes.
    Keywords: matplotlib.scale.InvertedLogTransform, invalidate
    """
    output_var = input_var.invalidate()

def matplotlib_scale_InvertedLogTransform_inverted(self: 'any') -> 'any_computed':
    r"""
    Return the corresponding inverse transformation.
    It holds ``x == self.inverted().transform(self.transform(x))``.
    The return value of this method should be treated as
    Keywords: matplotlib.scale.InvertedLogTransform, inverted
    """
    output_var = input_var.inverted()

def matplotlib_scale_InvertedLogTransform_set_children(self: 'any') -> 'any_computed':
    r"""
    Set the children of the transform, to let the invalidation
    system know which transforms can invalidate this transform.
    Should be called from the constructor of any transforms that
    depend on other transforms.
    Keywords: matplotlib.scale.InvertedLogTransform, set_children
    """
    output_var = input_var.set_children()

def matplotlib_scale_InvertedLogTransform_transform(self: 'any') -> 'any_computed':
    r"""
    Apply this transformation on the given array of *values*.
    Parameters
    ----------
    values : array-like
    Keywords: matplotlib.scale.InvertedLogTransform, transform
    """
    output_var = input_var.transform()

def matplotlib_scale_InvertedLogTransform_transform_affine(self: 'any') -> 'any_computed':
    r"""
    Apply only the affine part of this transformation on the
    given array of values.
    ``transform(values)`` is always equivalent to
    ``transform_affine(transform_non_affine(values))``.
    Keywords: matplotlib.scale.InvertedLogTransform, transform_affine
    """
    output_var = input_var.transform_affine()

def matplotlib_scale_InvertedLogTransform_transform_angles(self: 'any') -> 'any_computed':
    r"""
    Transform a set of angles anchored at specific locations.
    Parameters
    ----------
    angles : (N,) array-like
    Keywords: matplotlib.scale.InvertedLogTransform, transform_angles
    """
    output_var = input_var.transform_angles()

def matplotlib_scale_InvertedLogTransform_transform_bbox(self: 'any') -> 'any_computed':
    r"""
    Transform the given bounding box.
    For smarter transforms including caching (a common requirement in
    Matplotlib), see `TransformedBbox`.
    Keywords: matplotlib.scale.InvertedLogTransform, transform_bbox
    """
    output_var = input_var.transform_bbox()

def matplotlib_scale_InvertedLogTransform_transform_non_affine(self: 'any') -> 'any_computed':
    r"""
    Apply only the non-affine part of this transformation.
    ``transform(values)`` is always equivalent to
    ``transform_affine(transform_non_affine(values))``.
    Keywords: matplotlib.scale.InvertedLogTransform, transform_non_affine
    """
    output_var = input_var.transform_non_affine()

def matplotlib_scale_InvertedLogTransform_transform_path(self: 'any') -> 'any_computed':
    r"""
    Apply the transform to `.Path` *path*, returning a new `.Path`.
    In some cases, this transform may insert curves into the path
    that began as line segments.
    Keywords: matplotlib.scale.InvertedLogTransform, transform_path
    """
    output_var = input_var.transform_path()

def matplotlib_scale_InvertedLogTransform_transform_path_affine(self: 'any') -> 'any_computed':
    r"""
    Apply the affine part of this transform to `.Path` *path*, returning a
    new `.Path`.
    ``transform_path(path)`` is equivalent to
    ``transform_path_affine(transform_path_non_affine(values))``.
    Keywords: matplotlib.scale.InvertedLogTransform, transform_path_affine
    """
    output_var = input_var.transform_path_affine()

def matplotlib_scale_InvertedLogTransform_transform_path_non_affine(self: 'any') -> 'any_computed':
    r"""
    Apply the non-affine part of this transform to `.Path` *path*,
    returning a new `.Path`.
    ``transform_path(path)`` is equivalent to
    ``transform_path_affine(transform_path_non_affine(values))``.
    Keywords: matplotlib.scale.InvertedLogTransform, transform_path_non_affine
    """
    output_var = input_var.transform_path_non_affine()

def matplotlib_scale_InvertedLogTransform_transform_point(self: 'any') -> 'any_computed':
    r"""
    Return a transformed point.
    This function is only kept for backcompatibility; the more general
    `.transform` method is capable of transforming both a list of points
    and a single point.
    Keywords: matplotlib.scale.InvertedLogTransform, transform_point
    """
    output_var = input_var.transform_point()

def matplotlib_scale_InvertedSymmetricalLogTransform_contains_branch(self: 'any') -> 'any_computed':
    r"""
    Return whether the given transform is a sub-tree of this transform.
    This routine uses transform equality to identify sub-trees, therefore
    in many situations it is object id which will be used.
    Keywords: matplotlib.scale.InvertedSymmetricalLogTransform, contains_branch
    """
    output_var = input_var.contains_branch()

def matplotlib_scale_InvertedSymmetricalLogTransform_contains_branch_seperately(self: 'any') -> 'any_computed':
    r"""
    Return whether the given branch is a sub-tree of this transform on
    each separate dimension.
    A common use for this method is to identify if a transform is a blended
    transform containing an Axes' data transform. e.g.::
    Keywords: matplotlib.scale.InvertedSymmetricalLogTransform, contains_branch_seperately
    """
    output_var = input_var.contains_branch_seperately()

def matplotlib_scale_InvertedSymmetricalLogTransform_frozen(self: 'any') -> 'any_computed':
    r"""
    Return a frozen copy of this transform node.  The frozen copy will not
    be updated when its children change.  Useful for storing a previously
    known state of a transform where ``copy.deepcopy()`` might normally be
    used.
    Keywords: matplotlib.scale.InvertedSymmetricalLogTransform, frozen
    """
    output_var = input_var.frozen()

def matplotlib_scale_InvertedSymmetricalLogTransform_get_affine(self: 'any') -> 'any_computed':
    r"""
    Get the affine part of this transform.
    Keywords: matplotlib.scale.InvertedSymmetricalLogTransform, get_affine
    """
    output_var = input_var.get_affine()

def matplotlib_scale_InvertedSymmetricalLogTransform_get_matrix(self: 'any') -> 'any_computed':
    r"""
    Get the matrix for the affine part of this transform.
    Keywords: matplotlib.scale.InvertedSymmetricalLogTransform, get_matrix
    """
    output_var = input_var.get_matrix()

def matplotlib_scale_InvertedSymmetricalLogTransform_invalidate(self: 'any') -> 'any_computed':
    r"""
    Invalidate this `TransformNode` and triggers an invalidation of its
    ancestors.  Should be called any time the transform changes.
    Keywords: matplotlib.scale.InvertedSymmetricalLogTransform, invalidate
    """
    output_var = input_var.invalidate()

def matplotlib_scale_InvertedSymmetricalLogTransform_inverted(self: 'any') -> 'any_computed':
    r"""
    Return the corresponding inverse transformation.
    It holds ``x == self.inverted().transform(self.transform(x))``.
    The return value of this method should be treated as
    Keywords: matplotlib.scale.InvertedSymmetricalLogTransform, inverted
    """
    output_var = input_var.inverted()

def matplotlib_scale_InvertedSymmetricalLogTransform_set_children(self: 'any') -> 'any_computed':
    r"""
    Set the children of the transform, to let the invalidation
    system know which transforms can invalidate this transform.
    Should be called from the constructor of any transforms that
    depend on other transforms.
    Keywords: matplotlib.scale.InvertedSymmetricalLogTransform, set_children
    """
    output_var = input_var.set_children()

def matplotlib_scale_InvertedSymmetricalLogTransform_transform(self: 'any') -> 'any_computed':
    r"""
    Apply this transformation on the given array of *values*.
    Parameters
    ----------
    values : array-like
    Keywords: matplotlib.scale.InvertedSymmetricalLogTransform, transform
    """
    output_var = input_var.transform()

def matplotlib_scale_InvertedSymmetricalLogTransform_transform_affine(self: 'any') -> 'any_computed':
    r"""
    Apply only the affine part of this transformation on the
    given array of values.
    ``transform(values)`` is always equivalent to
    ``transform_affine(transform_non_affine(values))``.
    Keywords: matplotlib.scale.InvertedSymmetricalLogTransform, transform_affine
    """
    output_var = input_var.transform_affine()

def matplotlib_scale_InvertedSymmetricalLogTransform_transform_angles(self: 'any') -> 'any_computed':
    r"""
    Transform a set of angles anchored at specific locations.
    Parameters
    ----------
    angles : (N,) array-like
    Keywords: matplotlib.scale.InvertedSymmetricalLogTransform, transform_angles
    """
    output_var = input_var.transform_angles()

def matplotlib_scale_InvertedSymmetricalLogTransform_transform_bbox(self: 'any') -> 'any_computed':
    r"""
    Transform the given bounding box.
    For smarter transforms including caching (a common requirement in
    Matplotlib), see `TransformedBbox`.
    Keywords: matplotlib.scale.InvertedSymmetricalLogTransform, transform_bbox
    """
    output_var = input_var.transform_bbox()

def matplotlib_scale_InvertedSymmetricalLogTransform_transform_non_affine(self: 'any') -> 'any_computed':
    r"""
    Apply only the non-affine part of this transformation.
    ``transform(values)`` is always equivalent to
    ``transform_affine(transform_non_affine(values))``.
    Keywords: matplotlib.scale.InvertedSymmetricalLogTransform, transform_non_affine
    """
    output_var = input_var.transform_non_affine()

def matplotlib_scale_InvertedSymmetricalLogTransform_transform_path(self: 'any') -> 'any_computed':
    r"""
    Apply the transform to `.Path` *path*, returning a new `.Path`.
    In some cases, this transform may insert curves into the path
    that began as line segments.
    Keywords: matplotlib.scale.InvertedSymmetricalLogTransform, transform_path
    """
    output_var = input_var.transform_path()

def matplotlib_scale_InvertedSymmetricalLogTransform_transform_path_affine(self: 'any') -> 'any_computed':
    r"""
    Apply the affine part of this transform to `.Path` *path*, returning a
    new `.Path`.
    ``transform_path(path)`` is equivalent to
    ``transform_path_affine(transform_path_non_affine(values))``.
    Keywords: matplotlib.scale.InvertedSymmetricalLogTransform, transform_path_affine
    """
    output_var = input_var.transform_path_affine()

def matplotlib_scale_InvertedSymmetricalLogTransform_transform_path_non_affine(self: 'any') -> 'any_computed':
    r"""
    Apply the non-affine part of this transform to `.Path` *path*,
    returning a new `.Path`.
    ``transform_path(path)`` is equivalent to
    ``transform_path_affine(transform_path_non_affine(values))``.
    Keywords: matplotlib.scale.InvertedSymmetricalLogTransform, transform_path_non_affine
    """
    output_var = input_var.transform_path_non_affine()

def matplotlib_scale_InvertedSymmetricalLogTransform_transform_point(self: 'any') -> 'any_computed':
    r"""
    Return a transformed point.
    This function is only kept for backcompatibility; the more general
    `.transform` method is capable of transforming both a list of points
    and a single point.
    Keywords: matplotlib.scale.InvertedSymmetricalLogTransform, transform_point
    """
    output_var = input_var.transform_point()

def matplotlib_scale_LinearScale_get_transform(self: 'any') -> 'any_computed':
    r"""
    Return the transform for linear scaling, which is just the
    `~matplotlib.transforms.IdentityTransform`.
    Keywords: matplotlib.scale.LinearScale, get_transform
    """
    output_var = input_var.get_transform()

def matplotlib_scale_LinearScale_limit_range_for_scale(self: 'any') -> 'any_computed':
    r"""
    Return the range *vmin*, *vmax*, restricted to the
    domain supported by this scale (if any).
    *minpos* should be the minimum positive value in the data.
    This is used by log scales to determine a minimum value.
    Keywords: matplotlib.scale.LinearScale, limit_range_for_scale
    """
    output_var = input_var.limit_range_for_scale()

def matplotlib_scale_LinearScale_set_default_locators_and_formatters(self: 'any') -> 'any_computed':
    r"""
    Set the locators and formatters of *axis* to instances suitable for
    this scale.
    Keywords: matplotlib.scale.LinearScale, set_default_locators_and_formatters
    """
    output_var = input_var.set_default_locators_and_formatters()

def matplotlib_scale_LogFormatterSciNotation_create_dummy_axis(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.scale.LogFormatterSciNotation, create_dummy_axis
    """
    output_var = input_var.create_dummy_axis()

def matplotlib_scale_LogFormatterSciNotation_fix_minus(s: 'any') -> 'any_computed':
    r"""
    Some classes may want to replace a hyphen for minus with the proper
    Unicode symbol (U+2212) for typographical correctness.  This is a
    helper method to perform such a replacement when it is enabled via
    :rc:`axes.unicode_minus`.
    Keywords: matplotlib.scale.LogFormatterSciNotation, fix_minus
    """
    output_var = input_var.fix_minus()

def matplotlib_scale_LogFormatterSciNotation_format_data(self: 'any') -> 'any_computed':
    r"""
    Return the full string representation of the value with the
    position unspecified.
    Keywords: matplotlib.scale.LogFormatterSciNotation, format_data
    """
    output_var = input_var.format_data()

def matplotlib_scale_LogFormatterSciNotation_format_data_short(self: 'any') -> 'any_computed':
    r"""
    Return a short string version of the tick value.
    Defaults to the position-independent long value.
    Keywords: matplotlib.scale.LogFormatterSciNotation, format_data_short
    """
    output_var = input_var.format_data_short()

def matplotlib_scale_LogFormatterSciNotation_format_ticks(self: 'any') -> 'any_computed':
    r"""
    Return the tick labels for all the ticks at once.
    Keywords: matplotlib.scale.LogFormatterSciNotation, format_ticks
    """
    output_var = input_var.format_ticks()

def matplotlib_scale_LogFormatterSciNotation_get_offset(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.scale.LogFormatterSciNotation, get_offset
    """
    output_var = input_var.get_offset()

def matplotlib_scale_LogFormatterSciNotation_set_axis(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.scale.LogFormatterSciNotation, set_axis
    """
    output_var = input_var.set_axis()

def matplotlib_scale_LogFormatterSciNotation_set_base(self: 'any') -> 'any_computed':
    r"""
    Change the *base* for labeling.
    .. warning::
       Should always match the base used for :class:`LogLocator`
    Keywords: matplotlib.scale.LogFormatterSciNotation, set_base
    """
    output_var = input_var.set_base()

def matplotlib_scale_LogFormatterSciNotation_set_label_minor(self: 'any') -> 'any_computed':
    r"""
    Switch minor tick labeling on or off.
    Parameters
    ----------
    labelOnlyBase : bool
    Keywords: matplotlib.scale.LogFormatterSciNotation, set_label_minor
    """
    output_var = input_var.set_label_minor()

def matplotlib_scale_LogFormatterSciNotation_set_locs(self: 'any') -> 'any_computed':
    r"""
    Use axis view limits to control which ticks are labeled.
    The *locs* parameter is ignored in the present algorithm.
    Keywords: matplotlib.scale.LogFormatterSciNotation, set_locs
    """
    output_var = input_var.set_locs()

def matplotlib_scale_LogLocator_create_dummy_axis(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.scale.LogLocator, create_dummy_axis
    """
    output_var = input_var.create_dummy_axis()

def matplotlib_scale_LogLocator_nonsingular(self: 'any') -> 'any_computed':
    r"""
    Adjust a range as needed to avoid singularities.
    This method gets called during autoscaling, with ``(v0, v1)`` set to
    the data limits on the Axes if the Axes contains any data, or
    ``(-inf, +inf)`` if not.
    Keywords: matplotlib.scale.LogLocator, nonsingular
    """
    output_var = input_var.nonsingular()

def matplotlib_scale_LogLocator_raise_if_exceeds(self: 'any') -> 'any_computed':
    r"""
    Log at WARNING level if *locs* is longer than `Locator.MAXTICKS`.
    This is intended to be called immediately before returning *locs* from
    ``__call__`` to inform users in case their Locator returns a huge
    number of ticks, causing Matplotlib to run out of memory.
    Keywords: matplotlib.scale.LogLocator, raise_if_exceeds
    """
    output_var = input_var.raise_if_exceeds()

def matplotlib_scale_LogLocator_set_axis(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.scale.LogLocator, set_axis
    """
    output_var = input_var.set_axis()

def matplotlib_scale_LogLocator_set_params(self: 'any') -> 'any_computed':
    r"""
    Set parameters within this locator.
    Keywords: matplotlib.scale.LogLocator, set_params
    """
    output_var = input_var.set_params()

def matplotlib_scale_LogLocator_tick_values(self: 'any') -> 'any_computed':
    r"""
    Return the values of the located ticks given **vmin** and **vmax**.
    .. note::
        To get tick locations with the vmin and vmax values defined
        automatically for the associated ``axis`` simply call
    Keywords: matplotlib.scale.LogLocator, tick_values
    """
    output_var = input_var.tick_values()

def matplotlib_scale_LogLocator_view_limits(self: 'any') -> 'any_computed':
    r"""
    Try to choose the view limits intelligently.
    Keywords: matplotlib.scale.LogLocator, view_limits
    """
    output_var = input_var.view_limits()

def matplotlib_scale_LogScale_get_transform(self: 'any') -> 'any_computed':
    r"""
    Return the `.LogTransform` associated with this scale.
    Keywords: matplotlib.scale.LogScale, get_transform
    """
    output_var = input_var.get_transform()

def matplotlib_scale_LogScale_limit_range_for_scale(self: 'any') -> 'any_computed':
    r"""
    Limit the domain to positive values.
    Keywords: matplotlib.scale.LogScale, limit_range_for_scale
    """
    output_var = input_var.limit_range_for_scale()

def matplotlib_scale_LogScale_set_default_locators_and_formatters(self: 'any') -> 'any_computed':
    r"""
    Set the locators and formatters of *axis* to instances suitable for
    this scale.
    Keywords: matplotlib.scale.LogScale, set_default_locators_and_formatters
    """
    output_var = input_var.set_default_locators_and_formatters()

def matplotlib_scale_LogTransform_contains_branch(self: 'any') -> 'any_computed':
    r"""
    Return whether the given transform is a sub-tree of this transform.
    This routine uses transform equality to identify sub-trees, therefore
    in many situations it is object id which will be used.
    Keywords: matplotlib.scale.LogTransform, contains_branch
    """
    output_var = input_var.contains_branch()

def matplotlib_scale_LogTransform_contains_branch_seperately(self: 'any') -> 'any_computed':
    r"""
    Return whether the given branch is a sub-tree of this transform on
    each separate dimension.
    A common use for this method is to identify if a transform is a blended
    transform containing an Axes' data transform. e.g.::
    Keywords: matplotlib.scale.LogTransform, contains_branch_seperately
    """
    output_var = input_var.contains_branch_seperately()

def matplotlib_scale_LogTransform_frozen(self: 'any') -> 'any_computed':
    r"""
    Return a frozen copy of this transform node.  The frozen copy will not
    be updated when its children change.  Useful for storing a previously
    known state of a transform where ``copy.deepcopy()`` might normally be
    used.
    Keywords: matplotlib.scale.LogTransform, frozen
    """
    output_var = input_var.frozen()

def matplotlib_scale_LogTransform_get_affine(self: 'any') -> 'any_computed':
    r"""
    Get the affine part of this transform.
    Keywords: matplotlib.scale.LogTransform, get_affine
    """
    output_var = input_var.get_affine()

def matplotlib_scale_LogTransform_get_matrix(self: 'any') -> 'any_computed':
    r"""
    Get the matrix for the affine part of this transform.
    Keywords: matplotlib.scale.LogTransform, get_matrix
    """
    output_var = input_var.get_matrix()

def matplotlib_scale_LogTransform_invalidate(self: 'any') -> 'any_computed':
    r"""
    Invalidate this `TransformNode` and triggers an invalidation of its
    ancestors.  Should be called any time the transform changes.
    Keywords: matplotlib.scale.LogTransform, invalidate
    """
    output_var = input_var.invalidate()

def matplotlib_scale_LogTransform_inverted(self: 'any') -> 'any_computed':
    r"""
    Return the corresponding inverse transformation.
    It holds ``x == self.inverted().transform(self.transform(x))``.
    The return value of this method should be treated as
    Keywords: matplotlib.scale.LogTransform, inverted
    """
    output_var = input_var.inverted()

def matplotlib_scale_LogTransform_set_children(self: 'any') -> 'any_computed':
    r"""
    Set the children of the transform, to let the invalidation
    system know which transforms can invalidate this transform.
    Should be called from the constructor of any transforms that
    depend on other transforms.
    Keywords: matplotlib.scale.LogTransform, set_children
    """
    output_var = input_var.set_children()

def matplotlib_scale_LogTransform_transform(self: 'any') -> 'any_computed':
    r"""
    Apply this transformation on the given array of *values*.
    Parameters
    ----------
    values : array-like
    Keywords: matplotlib.scale.LogTransform, transform
    """
    output_var = input_var.transform()

def matplotlib_scale_LogTransform_transform_affine(self: 'any') -> 'any_computed':
    r"""
    Apply only the affine part of this transformation on the
    given array of values.
    ``transform(values)`` is always equivalent to
    ``transform_affine(transform_non_affine(values))``.
    Keywords: matplotlib.scale.LogTransform, transform_affine
    """
    output_var = input_var.transform_affine()

def matplotlib_scale_LogTransform_transform_angles(self: 'any') -> 'any_computed':
    r"""
    Transform a set of angles anchored at specific locations.
    Parameters
    ----------
    angles : (N,) array-like
    Keywords: matplotlib.scale.LogTransform, transform_angles
    """
    output_var = input_var.transform_angles()

def matplotlib_scale_LogTransform_transform_bbox(self: 'any') -> 'any_computed':
    r"""
    Transform the given bounding box.
    For smarter transforms including caching (a common requirement in
    Matplotlib), see `TransformedBbox`.
    Keywords: matplotlib.scale.LogTransform, transform_bbox
    """
    output_var = input_var.transform_bbox()

def matplotlib_scale_LogTransform_transform_non_affine(self: 'any') -> 'any_computed':
    r"""
    Apply only the non-affine part of this transformation.
    ``transform(values)`` is always equivalent to
    ``transform_affine(transform_non_affine(values))``.
    Keywords: matplotlib.scale.LogTransform, transform_non_affine
    """
    output_var = input_var.transform_non_affine()

def matplotlib_scale_LogTransform_transform_path(self: 'any') -> 'any_computed':
    r"""
    Apply the transform to `.Path` *path*, returning a new `.Path`.
    In some cases, this transform may insert curves into the path
    that began as line segments.
    Keywords: matplotlib.scale.LogTransform, transform_path
    """
    output_var = input_var.transform_path()

def matplotlib_scale_LogTransform_transform_path_affine(self: 'any') -> 'any_computed':
    r"""
    Apply the affine part of this transform to `.Path` *path*, returning a
    new `.Path`.
    ``transform_path(path)`` is equivalent to
    ``transform_path_affine(transform_path_non_affine(values))``.
    Keywords: matplotlib.scale.LogTransform, transform_path_affine
    """
    output_var = input_var.transform_path_affine()

def matplotlib_scale_LogTransform_transform_path_non_affine(self: 'any') -> 'any_computed':
    r"""
    Apply the non-affine part of this transform to `.Path` *path*,
    returning a new `.Path`.
    ``transform_path(path)`` is equivalent to
    ``transform_path_affine(transform_path_non_affine(values))``.
    Keywords: matplotlib.scale.LogTransform, transform_path_non_affine
    """
    output_var = input_var.transform_path_non_affine()

def matplotlib_scale_LogTransform_transform_point(self: 'any') -> 'any_computed':
    r"""
    Return a transformed point.
    This function is only kept for backcompatibility; the more general
    `.transform` method is capable of transforming both a list of points
    and a single point.
    Keywords: matplotlib.scale.LogTransform, transform_point
    """
    output_var = input_var.transform_point()

def matplotlib_scale_LogisticTransform_contains_branch(self: 'any') -> 'any_computed':
    r"""
    Return whether the given transform is a sub-tree of this transform.
    This routine uses transform equality to identify sub-trees, therefore
    in many situations it is object id which will be used.
    Keywords: matplotlib.scale.LogisticTransform, contains_branch
    """
    output_var = input_var.contains_branch()

def matplotlib_scale_LogisticTransform_contains_branch_seperately(self: 'any') -> 'any_computed':
    r"""
    Return whether the given branch is a sub-tree of this transform on
    each separate dimension.
    A common use for this method is to identify if a transform is a blended
    transform containing an Axes' data transform. e.g.::
    Keywords: matplotlib.scale.LogisticTransform, contains_branch_seperately
    """
    output_var = input_var.contains_branch_seperately()

def matplotlib_scale_LogisticTransform_frozen(self: 'any') -> 'any_computed':
    r"""
    Return a frozen copy of this transform node.  The frozen copy will not
    be updated when its children change.  Useful for storing a previously
    known state of a transform where ``copy.deepcopy()`` might normally be
    used.
    Keywords: matplotlib.scale.LogisticTransform, frozen
    """
    output_var = input_var.frozen()

def matplotlib_scale_LogisticTransform_get_affine(self: 'any') -> 'any_computed':
    r"""
    Get the affine part of this transform.
    Keywords: matplotlib.scale.LogisticTransform, get_affine
    """
    output_var = input_var.get_affine()

def matplotlib_scale_LogisticTransform_get_matrix(self: 'any') -> 'any_computed':
    r"""
    Get the matrix for the affine part of this transform.
    Keywords: matplotlib.scale.LogisticTransform, get_matrix
    """
    output_var = input_var.get_matrix()

def matplotlib_scale_LogisticTransform_invalidate(self: 'any') -> 'any_computed':
    r"""
    Invalidate this `TransformNode` and triggers an invalidation of its
    ancestors.  Should be called any time the transform changes.
    Keywords: matplotlib.scale.LogisticTransform, invalidate
    """
    output_var = input_var.invalidate()

def matplotlib_scale_LogisticTransform_inverted(self: 'any') -> 'any_computed':
    r"""
    Return the corresponding inverse transformation.
    It holds ``x == self.inverted().transform(self.transform(x))``.
    The return value of this method should be treated as
    Keywords: matplotlib.scale.LogisticTransform, inverted
    """
    output_var = input_var.inverted()

def matplotlib_scale_LogisticTransform_set_children(self: 'any') -> 'any_computed':
    r"""
    Set the children of the transform, to let the invalidation
    system know which transforms can invalidate this transform.
    Should be called from the constructor of any transforms that
    depend on other transforms.
    Keywords: matplotlib.scale.LogisticTransform, set_children
    """
    output_var = input_var.set_children()

def matplotlib_scale_LogisticTransform_transform(self: 'any') -> 'any_computed':
    r"""
    Apply this transformation on the given array of *values*.
    Parameters
    ----------
    values : array-like
    Keywords: matplotlib.scale.LogisticTransform, transform
    """
    output_var = input_var.transform()

def matplotlib_scale_LogisticTransform_transform_affine(self: 'any') -> 'any_computed':
    r"""
    Apply only the affine part of this transformation on the
    given array of values.
    ``transform(values)`` is always equivalent to
    ``transform_affine(transform_non_affine(values))``.
    Keywords: matplotlib.scale.LogisticTransform, transform_affine
    """
    output_var = input_var.transform_affine()

def matplotlib_scale_LogisticTransform_transform_angles(self: 'any') -> 'any_computed':
    r"""
    Transform a set of angles anchored at specific locations.
    Parameters
    ----------
    angles : (N,) array-like
    Keywords: matplotlib.scale.LogisticTransform, transform_angles
    """
    output_var = input_var.transform_angles()

def matplotlib_scale_LogisticTransform_transform_bbox(self: 'any') -> 'any_computed':
    r"""
    Transform the given bounding box.
    For smarter transforms including caching (a common requirement in
    Matplotlib), see `TransformedBbox`.
    Keywords: matplotlib.scale.LogisticTransform, transform_bbox
    """
    output_var = input_var.transform_bbox()

def matplotlib_scale_LogisticTransform_transform_non_affine(self: 'any') -> 'any_computed':
    r"""
    logistic transform (base 10)
    Keywords: matplotlib.scale.LogisticTransform, transform_non_affine
    """
    output_var = input_var.transform_non_affine()

def matplotlib_scale_LogisticTransform_transform_path(self: 'any') -> 'any_computed':
    r"""
    Apply the transform to `.Path` *path*, returning a new `.Path`.
    In some cases, this transform may insert curves into the path
    that began as line segments.
    Keywords: matplotlib.scale.LogisticTransform, transform_path
    """
    output_var = input_var.transform_path()

def matplotlib_scale_LogisticTransform_transform_path_affine(self: 'any') -> 'any_computed':
    r"""
    Apply the affine part of this transform to `.Path` *path*, returning a
    new `.Path`.
    ``transform_path(path)`` is equivalent to
    ``transform_path_affine(transform_path_non_affine(values))``.
    Keywords: matplotlib.scale.LogisticTransform, transform_path_affine
    """
    output_var = input_var.transform_path_affine()

def matplotlib_scale_LogisticTransform_transform_path_non_affine(self: 'any') -> 'any_computed':
    r"""
    Apply the non-affine part of this transform to `.Path` *path*,
    returning a new `.Path`.
    ``transform_path(path)`` is equivalent to
    ``transform_path_affine(transform_path_non_affine(values))``.
    Keywords: matplotlib.scale.LogisticTransform, transform_path_non_affine
    """
    output_var = input_var.transform_path_non_affine()

def matplotlib_scale_LogisticTransform_transform_point(self: 'any') -> 'any_computed':
    r"""
    Return a transformed point.
    This function is only kept for backcompatibility; the more general
    `.transform` method is capable of transforming both a list of points
    and a single point.
    Keywords: matplotlib.scale.LogisticTransform, transform_point
    """
    output_var = input_var.transform_point()

def matplotlib_scale_LogitFormatter_create_dummy_axis(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.scale.LogitFormatter, create_dummy_axis
    """
    output_var = input_var.create_dummy_axis()

def matplotlib_scale_LogitFormatter_fix_minus(s: 'any') -> 'any_computed':
    r"""
    Some classes may want to replace a hyphen for minus with the proper
    Unicode symbol (U+2212) for typographical correctness.  This is a
    helper method to perform such a replacement when it is enabled via
    :rc:`axes.unicode_minus`.
    Keywords: matplotlib.scale.LogitFormatter, fix_minus
    """
    output_var = input_var.fix_minus()

def matplotlib_scale_LogitFormatter_format_data(self: 'any') -> 'any_computed':
    r"""
    Return the full string representation of the value with the
    position unspecified.
    Keywords: matplotlib.scale.LogitFormatter, format_data
    """
    output_var = input_var.format_data()

def matplotlib_scale_LogitFormatter_format_data_short(self: 'any') -> 'any_computed':
    r"""
    Return a short string version of the tick value.
    Defaults to the position-independent long value.
    Keywords: matplotlib.scale.LogitFormatter, format_data_short
    """
    output_var = input_var.format_data_short()

def matplotlib_scale_LogitFormatter_format_ticks(self: 'any') -> 'any_computed':
    r"""
    Return the tick labels for all the ticks at once.
    Keywords: matplotlib.scale.LogitFormatter, format_ticks
    """
    output_var = input_var.format_ticks()

def matplotlib_scale_LogitFormatter_get_offset(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.scale.LogitFormatter, get_offset
    """
    output_var = input_var.get_offset()

def matplotlib_scale_LogitFormatter_set_axis(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.scale.LogitFormatter, set_axis
    """
    output_var = input_var.set_axis()

def matplotlib_scale_LogitFormatter_set_locs(self: 'any') -> 'any_computed':
    r"""
    Set the locations of the ticks.
    This method is called before computing the tick labels because some
    formatters need to know all tick locations to do so.
    Keywords: matplotlib.scale.LogitFormatter, set_locs
    """
    output_var = input_var.set_locs()

def matplotlib_scale_LogitFormatter_set_minor_number(self: 'any') -> 'any_computed':
    r"""
    Set the number of minor ticks to label when some minor ticks are
    labelled.
    Parameters
    ----------
    Keywords: matplotlib.scale.LogitFormatter, set_minor_number
    """
    output_var = input_var.set_minor_number()

def matplotlib_scale_LogitFormatter_set_minor_threshold(self: 'any') -> 'any_computed':
    r"""
    Set the threshold for labelling minors ticks.
    Parameters
    ----------
    minor_threshold : int
    Keywords: matplotlib.scale.LogitFormatter, set_minor_threshold
    """
    output_var = input_var.set_minor_threshold()

def matplotlib_scale_LogitFormatter_set_one_half(self: 'any') -> 'any_computed':
    r"""
    Set the way one half is displayed.
    one_half : str
        The string used to represent 1/2.
    Keywords: matplotlib.scale.LogitFormatter, set_one_half
    """
    output_var = input_var.set_one_half()

def matplotlib_scale_LogitFormatter_use_overline(self: 'any') -> 'any_computed':
    r"""
    Switch display mode with overline for labelling p>1/2.
    Parameters
    ----------
    use_overline : bool
    Keywords: matplotlib.scale.LogitFormatter, use_overline
    """
    output_var = input_var.use_overline()

def matplotlib_scale_LogitLocator_create_dummy_axis(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.scale.LogitLocator, create_dummy_axis
    """
    output_var = input_var.create_dummy_axis()

def matplotlib_scale_LogitLocator_nonsingular(self: 'any') -> 'any_computed':
    r"""
    Adjust a range as needed to avoid singularities.
    This method gets called during autoscaling, with ``(v0, v1)`` set to
    the data limits on the Axes if the Axes contains any data, or
    ``(-inf, +inf)`` if not.
    Keywords: matplotlib.scale.LogitLocator, nonsingular
    """
    output_var = input_var.nonsingular()

def matplotlib_scale_LogitLocator_raise_if_exceeds(self: 'any') -> 'any_computed':
    r"""
    Log at WARNING level if *locs* is longer than `Locator.MAXTICKS`.
    This is intended to be called immediately before returning *locs* from
    ``__call__`` to inform users in case their Locator returns a huge
    number of ticks, causing Matplotlib to run out of memory.
    Keywords: matplotlib.scale.LogitLocator, raise_if_exceeds
    """
    output_var = input_var.raise_if_exceeds()

def matplotlib_scale_LogitLocator_set_axis(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.scale.LogitLocator, set_axis
    """
    output_var = input_var.set_axis()

def matplotlib_scale_LogitLocator_set_params(self: 'any') -> 'any_computed':
    r"""
    Set parameters within this locator.
    Keywords: matplotlib.scale.LogitLocator, set_params
    """
    output_var = input_var.set_params()

def matplotlib_scale_LogitLocator_tick_values(self: 'any') -> 'any_computed':
    r"""
    Return the values of the located ticks given **vmin** and **vmax**.
    .. note::
        To get tick locations with the vmin and vmax values defined
        automatically for the associated ``axis`` simply call
    Keywords: matplotlib.scale.LogitLocator, tick_values
    """
    output_var = input_var.tick_values()

def matplotlib_scale_LogitLocator_view_limits(self: 'any') -> 'any_computed':
    r"""
    Select a scale for the range from vmin to vmax.
    Subclasses should override this method to change locator behaviour.
    Keywords: matplotlib.scale.LogitLocator, view_limits
    """
    output_var = input_var.view_limits()

def matplotlib_scale_LogitScale_get_transform(self: 'any') -> 'any_computed':
    r"""
    Return the `.LogitTransform` associated with this scale.
    Keywords: matplotlib.scale.LogitScale, get_transform
    """
    output_var = input_var.get_transform()

def matplotlib_scale_LogitScale_limit_range_for_scale(self: 'any') -> 'any_computed':
    r"""
    Limit the domain to values between 0 and 1 (excluded).
    Keywords: matplotlib.scale.LogitScale, limit_range_for_scale
    """
    output_var = input_var.limit_range_for_scale()

def matplotlib_scale_LogitScale_set_default_locators_and_formatters(self: 'any') -> 'any_computed':
    r"""
    Set the locators and formatters of *axis* to instances suitable for
    this scale.
    Keywords: matplotlib.scale.LogitScale, set_default_locators_and_formatters
    """
    output_var = input_var.set_default_locators_and_formatters()

def matplotlib_scale_LogitTransform_contains_branch(self: 'any') -> 'any_computed':
    r"""
    Return whether the given transform is a sub-tree of this transform.
    This routine uses transform equality to identify sub-trees, therefore
    in many situations it is object id which will be used.
    Keywords: matplotlib.scale.LogitTransform, contains_branch
    """
    output_var = input_var.contains_branch()

def matplotlib_scale_LogitTransform_contains_branch_seperately(self: 'any') -> 'any_computed':
    r"""
    Return whether the given branch is a sub-tree of this transform on
    each separate dimension.
    A common use for this method is to identify if a transform is a blended
    transform containing an Axes' data transform. e.g.::
    Keywords: matplotlib.scale.LogitTransform, contains_branch_seperately
    """
    output_var = input_var.contains_branch_seperately()

def matplotlib_scale_LogitTransform_frozen(self: 'any') -> 'any_computed':
    r"""
    Return a frozen copy of this transform node.  The frozen copy will not
    be updated when its children change.  Useful for storing a previously
    known state of a transform where ``copy.deepcopy()`` might normally be
    used.
    Keywords: matplotlib.scale.LogitTransform, frozen
    """
    output_var = input_var.frozen()

def matplotlib_scale_LogitTransform_get_affine(self: 'any') -> 'any_computed':
    r"""
    Get the affine part of this transform.
    Keywords: matplotlib.scale.LogitTransform, get_affine
    """
    output_var = input_var.get_affine()

def matplotlib_scale_LogitTransform_get_matrix(self: 'any') -> 'any_computed':
    r"""
    Get the matrix for the affine part of this transform.
    Keywords: matplotlib.scale.LogitTransform, get_matrix
    """
    output_var = input_var.get_matrix()

def matplotlib_scale_LogitTransform_invalidate(self: 'any') -> 'any_computed':
    r"""
    Invalidate this `TransformNode` and triggers an invalidation of its
    ancestors.  Should be called any time the transform changes.
    Keywords: matplotlib.scale.LogitTransform, invalidate
    """
    output_var = input_var.invalidate()

def matplotlib_scale_LogitTransform_inverted(self: 'any') -> 'any_computed':
    r"""
    Return the corresponding inverse transformation.
    It holds ``x == self.inverted().transform(self.transform(x))``.
    The return value of this method should be treated as
    Keywords: matplotlib.scale.LogitTransform, inverted
    """
    output_var = input_var.inverted()

def matplotlib_scale_LogitTransform_set_children(self: 'any') -> 'any_computed':
    r"""
    Set the children of the transform, to let the invalidation
    system know which transforms can invalidate this transform.
    Should be called from the constructor of any transforms that
    depend on other transforms.
    Keywords: matplotlib.scale.LogitTransform, set_children
    """
    output_var = input_var.set_children()

def matplotlib_scale_LogitTransform_transform(self: 'any') -> 'any_computed':
    r"""
    Apply this transformation on the given array of *values*.
    Parameters
    ----------
    values : array-like
    Keywords: matplotlib.scale.LogitTransform, transform
    """
    output_var = input_var.transform()

def matplotlib_scale_LogitTransform_transform_affine(self: 'any') -> 'any_computed':
    r"""
    Apply only the affine part of this transformation on the
    given array of values.
    ``transform(values)`` is always equivalent to
    ``transform_affine(transform_non_affine(values))``.
    Keywords: matplotlib.scale.LogitTransform, transform_affine
    """
    output_var = input_var.transform_affine()

def matplotlib_scale_LogitTransform_transform_angles(self: 'any') -> 'any_computed':
    r"""
    Transform a set of angles anchored at specific locations.
    Parameters
    ----------
    angles : (N,) array-like
    Keywords: matplotlib.scale.LogitTransform, transform_angles
    """
    output_var = input_var.transform_angles()

def matplotlib_scale_LogitTransform_transform_bbox(self: 'any') -> 'any_computed':
    r"""
    Transform the given bounding box.
    For smarter transforms including caching (a common requirement in
    Matplotlib), see `TransformedBbox`.
    Keywords: matplotlib.scale.LogitTransform, transform_bbox
    """
    output_var = input_var.transform_bbox()

def matplotlib_scale_LogitTransform_transform_non_affine(self: 'any') -> 'any_computed':
    r"""
    logit transform (base 10), masked or clipped
    Keywords: matplotlib.scale.LogitTransform, transform_non_affine
    """
    output_var = input_var.transform_non_affine()

def matplotlib_scale_LogitTransform_transform_path(self: 'any') -> 'any_computed':
    r"""
    Apply the transform to `.Path` *path*, returning a new `.Path`.
    In some cases, this transform may insert curves into the path
    that began as line segments.
    Keywords: matplotlib.scale.LogitTransform, transform_path
    """
    output_var = input_var.transform_path()

def matplotlib_scale_LogitTransform_transform_path_affine(self: 'any') -> 'any_computed':
    r"""
    Apply the affine part of this transform to `.Path` *path*, returning a
    new `.Path`.
    ``transform_path(path)`` is equivalent to
    ``transform_path_affine(transform_path_non_affine(values))``.
    Keywords: matplotlib.scale.LogitTransform, transform_path_affine
    """
    output_var = input_var.transform_path_affine()

def matplotlib_scale_LogitTransform_transform_path_non_affine(self: 'any') -> 'any_computed':
    r"""
    Apply the non-affine part of this transform to `.Path` *path*,
    returning a new `.Path`.
    ``transform_path(path)`` is equivalent to
    ``transform_path_affine(transform_path_non_affine(values))``.
    Keywords: matplotlib.scale.LogitTransform, transform_path_non_affine
    """
    output_var = input_var.transform_path_non_affine()

def matplotlib_scale_LogitTransform_transform_point(self: 'any') -> 'any_computed':
    r"""
    Return a transformed point.
    This function is only kept for backcompatibility; the more general
    `.transform` method is capable of transforming both a list of points
    and a single point.
    Keywords: matplotlib.scale.LogitTransform, transform_point
    """
    output_var = input_var.transform_point()

def matplotlib_scale_NullFormatter_create_dummy_axis(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.scale.NullFormatter, create_dummy_axis
    """
    output_var = input_var.create_dummy_axis()

def matplotlib_scale_NullFormatter_fix_minus(s: 'any') -> 'any_computed':
    r"""
    Some classes may want to replace a hyphen for minus with the proper
    Unicode symbol (U+2212) for typographical correctness.  This is a
    helper method to perform such a replacement when it is enabled via
    :rc:`axes.unicode_minus`.
    Keywords: matplotlib.scale.NullFormatter, fix_minus
    """
    output_var = input_var.fix_minus()

def matplotlib_scale_NullFormatter_format_data(self: 'any') -> 'any_computed':
    r"""
    Return the full string representation of the value with the
    position unspecified.
    Keywords: matplotlib.scale.NullFormatter, format_data
    """
    output_var = input_var.format_data()

def matplotlib_scale_NullFormatter_format_data_short(self: 'any') -> 'any_computed':
    r"""
    Return a short string version of the tick value.
    Defaults to the position-independent long value.
    Keywords: matplotlib.scale.NullFormatter, format_data_short
    """
    output_var = input_var.format_data_short()

def matplotlib_scale_NullFormatter_format_ticks(self: 'any') -> 'any_computed':
    r"""
    Return the tick labels for all the ticks at once.
    Keywords: matplotlib.scale.NullFormatter, format_ticks
    """
    output_var = input_var.format_ticks()

def matplotlib_scale_NullFormatter_get_offset(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.scale.NullFormatter, get_offset
    """
    output_var = input_var.get_offset()

def matplotlib_scale_NullFormatter_set_axis(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.scale.NullFormatter, set_axis
    """
    output_var = input_var.set_axis()

def matplotlib_scale_NullFormatter_set_locs(self: 'any') -> 'any_computed':
    r"""
    Set the locations of the ticks.
    This method is called before computing the tick labels because some
    formatters need to know all tick locations to do so.
    Keywords: matplotlib.scale.NullFormatter, set_locs
    """
    output_var = input_var.set_locs()

def matplotlib_scale_NullLocator_create_dummy_axis(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.scale.NullLocator, create_dummy_axis
    """
    output_var = input_var.create_dummy_axis()

def matplotlib_scale_NullLocator_nonsingular(self: 'any') -> 'any_computed':
    r"""
    Adjust a range as needed to avoid singularities.
    This method gets called during autoscaling, with ``(v0, v1)`` set to
    the data limits on the Axes if the Axes contains any data, or
    ``(-inf, +inf)`` if not.
    Keywords: matplotlib.scale.NullLocator, nonsingular
    """
    output_var = input_var.nonsingular()

def matplotlib_scale_NullLocator_raise_if_exceeds(self: 'any') -> 'any_computed':
    r"""
    Log at WARNING level if *locs* is longer than `Locator.MAXTICKS`.
    This is intended to be called immediately before returning *locs* from
    ``__call__`` to inform users in case their Locator returns a huge
    number of ticks, causing Matplotlib to run out of memory.
    Keywords: matplotlib.scale.NullLocator, raise_if_exceeds
    """
    output_var = input_var.raise_if_exceeds()

def matplotlib_scale_NullLocator_set_axis(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.scale.NullLocator, set_axis
    """
    output_var = input_var.set_axis()

def matplotlib_scale_NullLocator_set_params(self: 'any') -> 'any_computed':
    r"""
    Do nothing, and raise a warning. Any locator class not supporting the
    set_params() function will call this.
    Keywords: matplotlib.scale.NullLocator, set_params
    """
    output_var = input_var.set_params()

def matplotlib_scale_NullLocator_tick_values(self: 'any') -> 'any_computed':
    r"""
    Return the locations of the ticks.
    .. note::
        Because the values are Null, vmin and vmax are not used in this
    Keywords: matplotlib.scale.NullLocator, tick_values
    """
    output_var = input_var.tick_values()

def matplotlib_scale_NullLocator_view_limits(self: 'any') -> 'any_computed':
    r"""
    Select a scale for the range from vmin to vmax.
    Subclasses should override this method to change locator behaviour.
    Keywords: matplotlib.scale.NullLocator, view_limits
    """
    output_var = input_var.view_limits()

def matplotlib_scale_ScalarFormatter_create_dummy_axis(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.scale.ScalarFormatter, create_dummy_axis
    """
    output_var = input_var.create_dummy_axis()

def matplotlib_scale_ScalarFormatter_fix_minus(s: 'any') -> 'any_computed':
    r"""
    Some classes may want to replace a hyphen for minus with the proper
    Unicode symbol (U+2212) for typographical correctness.  This is a
    helper method to perform such a replacement when it is enabled via
    :rc:`axes.unicode_minus`.
    Keywords: matplotlib.scale.ScalarFormatter, fix_minus
    """
    output_var = input_var.fix_minus()

def matplotlib_scale_ScalarFormatter_format_data(self: 'any') -> 'any_computed':
    r"""
    Return the full string representation of the value with the
    position unspecified.
    Keywords: matplotlib.scale.ScalarFormatter, format_data
    """
    output_var = input_var.format_data()

def matplotlib_scale_ScalarFormatter_format_data_short(self: 'any') -> 'any_computed':
    r"""
    Return a short string version of the tick value.
    Defaults to the position-independent long value.
    Keywords: matplotlib.scale.ScalarFormatter, format_data_short
    """
    output_var = input_var.format_data_short()

def matplotlib_scale_ScalarFormatter_format_ticks(self: 'any') -> 'any_computed':
    r"""
    Return the tick labels for all the ticks at once.
    Keywords: matplotlib.scale.ScalarFormatter, format_ticks
    """
    output_var = input_var.format_ticks()

def matplotlib_scale_ScalarFormatter_get_offset(self: 'any') -> 'any_computed':
    r"""
    Return scientific notation, plus offset.
    Keywords: matplotlib.scale.ScalarFormatter, get_offset
    """
    output_var = input_var.get_offset()

def matplotlib_scale_ScalarFormatter_get_useLocale(self: 'any') -> 'any_computed':
    r"""
    Return whether locale settings are used for formatting.
    See Also
    --------
    ScalarFormatter.set_useLocale
    Keywords: matplotlib.scale.ScalarFormatter, get_useLocale
    """
    output_var = input_var.get_useLocale()

def matplotlib_scale_ScalarFormatter_get_useMathText(self: 'any') -> 'any_computed':
    r"""
    Return whether to use fancy math formatting.
    See Also
    --------
    ScalarFormatter.set_useMathText
    Keywords: matplotlib.scale.ScalarFormatter, get_useMathText
    """
    output_var = input_var.get_useMathText()

def matplotlib_scale_ScalarFormatter_get_useOffset(self: 'any') -> 'any_computed':
    r"""
    Return whether automatic mode for offset notation is active.
    This returns True if ``set_useOffset(True)``; it returns False if an
    explicit offset was set, e.g. ``set_useOffset(1000)``.
    Keywords: matplotlib.scale.ScalarFormatter, get_useOffset
    """
    output_var = input_var.get_useOffset()

def matplotlib_scale_ScalarFormatter_get_usetex(self: 'any') -> 'any_computed':
    r"""
    Return whether TeX's math mode is enabled for rendering.
    Keywords: matplotlib.scale.ScalarFormatter, get_usetex
    """
    output_var = input_var.get_usetex()

def matplotlib_scale_ScalarFormatter_set_axis(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.scale.ScalarFormatter, set_axis
    """
    output_var = input_var.set_axis()

def matplotlib_scale_ScalarFormatter_set_locs(self: 'any') -> 'any_computed':
    r"""
    Set the locations of the ticks.
    This method is called before computing the tick labels because some
    formatters need to know all tick locations to do so.
    Keywords: matplotlib.scale.ScalarFormatter, set_locs
    """
    output_var = input_var.set_locs()

def matplotlib_scale_ScalarFormatter_set_powerlimits(self: 'any') -> 'any_computed':
    r"""
    Set size thresholds for scientific notation.
    Parameters
    ----------
    lims : (int, int)
    Keywords: matplotlib.scale.ScalarFormatter, set_powerlimits
    """
    output_var = input_var.set_powerlimits()

def matplotlib_scale_ScalarFormatter_set_scientific(self: 'any') -> 'any_computed':
    r"""
    Turn scientific notation on or off.
    See Also
    --------
    ScalarFormatter.set_powerlimits
    Keywords: matplotlib.scale.ScalarFormatter, set_scientific
    """
    output_var = input_var.set_scientific()

def matplotlib_scale_ScalarFormatter_set_useLocale(self: 'any') -> 'any_computed':
    r"""
    Set whether to use locale settings for decimal sign and positive sign.
    Parameters
    ----------
    val : bool or None
    Keywords: matplotlib.scale.ScalarFormatter, set_useLocale
    """
    output_var = input_var.set_useLocale()

def matplotlib_scale_ScalarFormatter_set_useMathText(self: 'any') -> 'any_computed':
    r"""
    Set whether to use fancy math formatting.
    If active, scientific notation is formatted as :math:`1.2 \times 10^3`.
    Parameters
    Keywords: matplotlib.scale.ScalarFormatter, set_useMathText
    """
    output_var = input_var.set_useMathText()

def matplotlib_scale_ScalarFormatter_set_useOffset(self: 'any') -> 'any_computed':
    r"""
    Set whether to use offset notation.
    When formatting a set numbers whose value is large compared to their
    range, the formatter can separate an additive constant. This can
    shorten the formatted numbers so that they are less likely to overlap
    Keywords: matplotlib.scale.ScalarFormatter, set_useOffset
    """
    output_var = input_var.set_useOffset()

def matplotlib_scale_ScalarFormatter_set_usetex(self: 'any') -> 'any_computed':
    r"""
    Set whether to use TeX's math mode for rendering numbers in the formatter.
    Keywords: matplotlib.scale.ScalarFormatter, set_usetex
    """
    output_var = input_var.set_usetex()

def matplotlib_scale_ScaleBase_get_transform(self: 'any') -> 'any_computed':
    r"""
    Return the `.Transform` object associated with this scale.
    Keywords: matplotlib.scale.ScaleBase, get_transform
    """
    output_var = input_var.get_transform()

def matplotlib_scale_ScaleBase_limit_range_for_scale(self: 'any') -> 'any_computed':
    r"""
    Return the range *vmin*, *vmax*, restricted to the
    domain supported by this scale (if any).
    *minpos* should be the minimum positive value in the data.
    This is used by log scales to determine a minimum value.
    Keywords: matplotlib.scale.ScaleBase, limit_range_for_scale
    """
    output_var = input_var.limit_range_for_scale()

def matplotlib_scale_ScaleBase_set_default_locators_and_formatters(self: 'any') -> 'any_computed':
    r"""
    Set the locators and formatters of *axis* to instances suitable for
    this scale.
    Keywords: matplotlib.scale.ScaleBase, set_default_locators_and_formatters
    """
    output_var = input_var.set_default_locators_and_formatters()

def matplotlib_scale_SymmetricalLogLocator_create_dummy_axis(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.scale.SymmetricalLogLocator, create_dummy_axis
    """
    output_var = input_var.create_dummy_axis()

def matplotlib_scale_SymmetricalLogLocator_nonsingular(self: 'any') -> 'any_computed':
    r"""
    Adjust a range as needed to avoid singularities.
    This method gets called during autoscaling, with ``(v0, v1)`` set to
    the data limits on the Axes if the Axes contains any data, or
    ``(-inf, +inf)`` if not.
    Keywords: matplotlib.scale.SymmetricalLogLocator, nonsingular
    """
    output_var = input_var.nonsingular()

def matplotlib_scale_SymmetricalLogLocator_raise_if_exceeds(self: 'any') -> 'any_computed':
    r"""
    Log at WARNING level if *locs* is longer than `Locator.MAXTICKS`.
    This is intended to be called immediately before returning *locs* from
    ``__call__`` to inform users in case their Locator returns a huge
    number of ticks, causing Matplotlib to run out of memory.
    Keywords: matplotlib.scale.SymmetricalLogLocator, raise_if_exceeds
    """
    output_var = input_var.raise_if_exceeds()

def matplotlib_scale_SymmetricalLogLocator_set_axis(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.scale.SymmetricalLogLocator, set_axis
    """
    output_var = input_var.set_axis()

def matplotlib_scale_SymmetricalLogLocator_set_params(self: 'any') -> 'any_computed':
    r"""
    Set parameters within this locator.
    Keywords: matplotlib.scale.SymmetricalLogLocator, set_params
    """
    output_var = input_var.set_params()

def matplotlib_scale_SymmetricalLogLocator_tick_values(self: 'any') -> 'any_computed':
    r"""
    Return the values of the located ticks given **vmin** and **vmax**.
    .. note::
        To get tick locations with the vmin and vmax values defined
        automatically for the associated ``axis`` simply call
    Keywords: matplotlib.scale.SymmetricalLogLocator, tick_values
    """
    output_var = input_var.tick_values()

def matplotlib_scale_SymmetricalLogLocator_view_limits(self: 'any') -> 'any_computed':
    r"""
    Try to choose the view limits intelligently.
    Keywords: matplotlib.scale.SymmetricalLogLocator, view_limits
    """
    output_var = input_var.view_limits()

def matplotlib_scale_SymmetricalLogScale_get_transform(self: 'any') -> 'any_computed':
    r"""
    Return the `.SymmetricalLogTransform` associated with this scale.
    Keywords: matplotlib.scale.SymmetricalLogScale, get_transform
    """
    output_var = input_var.get_transform()

def matplotlib_scale_SymmetricalLogScale_limit_range_for_scale(self: 'any') -> 'any_computed':
    r"""
    Return the range *vmin*, *vmax*, restricted to the
    domain supported by this scale (if any).
    *minpos* should be the minimum positive value in the data.
    This is used by log scales to determine a minimum value.
    Keywords: matplotlib.scale.SymmetricalLogScale, limit_range_for_scale
    """
    output_var = input_var.limit_range_for_scale()

def matplotlib_scale_SymmetricalLogScale_set_default_locators_and_formatters(self: 'any') -> 'any_computed':
    r"""
    Set the locators and formatters of *axis* to instances suitable for
    this scale.
    Keywords: matplotlib.scale.SymmetricalLogScale, set_default_locators_and_formatters
    """
    output_var = input_var.set_default_locators_and_formatters()

def matplotlib_scale_SymmetricalLogTransform_contains_branch(self: 'any') -> 'any_computed':
    r"""
    Return whether the given transform is a sub-tree of this transform.
    This routine uses transform equality to identify sub-trees, therefore
    in many situations it is object id which will be used.
    Keywords: matplotlib.scale.SymmetricalLogTransform, contains_branch
    """
    output_var = input_var.contains_branch()

def matplotlib_scale_SymmetricalLogTransform_contains_branch_seperately(self: 'any') -> 'any_computed':
    r"""
    Return whether the given branch is a sub-tree of this transform on
    each separate dimension.
    A common use for this method is to identify if a transform is a blended
    transform containing an Axes' data transform. e.g.::
    Keywords: matplotlib.scale.SymmetricalLogTransform, contains_branch_seperately
    """
    output_var = input_var.contains_branch_seperately()

def matplotlib_scale_SymmetricalLogTransform_frozen(self: 'any') -> 'any_computed':
    r"""
    Return a frozen copy of this transform node.  The frozen copy will not
    be updated when its children change.  Useful for storing a previously
    known state of a transform where ``copy.deepcopy()`` might normally be
    used.
    Keywords: matplotlib.scale.SymmetricalLogTransform, frozen
    """
    output_var = input_var.frozen()

def matplotlib_scale_SymmetricalLogTransform_get_affine(self: 'any') -> 'any_computed':
    r"""
    Get the affine part of this transform.
    Keywords: matplotlib.scale.SymmetricalLogTransform, get_affine
    """
    output_var = input_var.get_affine()

def matplotlib_scale_SymmetricalLogTransform_get_matrix(self: 'any') -> 'any_computed':
    r"""
    Get the matrix for the affine part of this transform.
    Keywords: matplotlib.scale.SymmetricalLogTransform, get_matrix
    """
    output_var = input_var.get_matrix()

def matplotlib_scale_SymmetricalLogTransform_invalidate(self: 'any') -> 'any_computed':
    r"""
    Invalidate this `TransformNode` and triggers an invalidation of its
    ancestors.  Should be called any time the transform changes.
    Keywords: matplotlib.scale.SymmetricalLogTransform, invalidate
    """
    output_var = input_var.invalidate()

def matplotlib_scale_SymmetricalLogTransform_inverted(self: 'any') -> 'any_computed':
    r"""
    Return the corresponding inverse transformation.
    It holds ``x == self.inverted().transform(self.transform(x))``.
    The return value of this method should be treated as
    Keywords: matplotlib.scale.SymmetricalLogTransform, inverted
    """
    output_var = input_var.inverted()

def matplotlib_scale_SymmetricalLogTransform_set_children(self: 'any') -> 'any_computed':
    r"""
    Set the children of the transform, to let the invalidation
    system know which transforms can invalidate this transform.
    Should be called from the constructor of any transforms that
    depend on other transforms.
    Keywords: matplotlib.scale.SymmetricalLogTransform, set_children
    """
    output_var = input_var.set_children()

def matplotlib_scale_SymmetricalLogTransform_transform(self: 'any') -> 'any_computed':
    r"""
    Apply this transformation on the given array of *values*.
    Parameters
    ----------
    values : array-like
    Keywords: matplotlib.scale.SymmetricalLogTransform, transform
    """
    output_var = input_var.transform()

def matplotlib_scale_SymmetricalLogTransform_transform_affine(self: 'any') -> 'any_computed':
    r"""
    Apply only the affine part of this transformation on the
    given array of values.
    ``transform(values)`` is always equivalent to
    ``transform_affine(transform_non_affine(values))``.
    Keywords: matplotlib.scale.SymmetricalLogTransform, transform_affine
    """
    output_var = input_var.transform_affine()

def matplotlib_scale_SymmetricalLogTransform_transform_angles(self: 'any') -> 'any_computed':
    r"""
    Transform a set of angles anchored at specific locations.
    Parameters
    ----------
    angles : (N,) array-like
    Keywords: matplotlib.scale.SymmetricalLogTransform, transform_angles
    """
    output_var = input_var.transform_angles()

def matplotlib_scale_SymmetricalLogTransform_transform_bbox(self: 'any') -> 'any_computed':
    r"""
    Transform the given bounding box.
    For smarter transforms including caching (a common requirement in
    Matplotlib), see `TransformedBbox`.
    Keywords: matplotlib.scale.SymmetricalLogTransform, transform_bbox
    """
    output_var = input_var.transform_bbox()

def matplotlib_scale_SymmetricalLogTransform_transform_non_affine(self: 'any') -> 'any_computed':
    r"""
    Apply only the non-affine part of this transformation.
    ``transform(values)`` is always equivalent to
    ``transform_affine(transform_non_affine(values))``.
    Keywords: matplotlib.scale.SymmetricalLogTransform, transform_non_affine
    """
    output_var = input_var.transform_non_affine()

def matplotlib_scale_SymmetricalLogTransform_transform_path(self: 'any') -> 'any_computed':
    r"""
    Apply the transform to `.Path` *path*, returning a new `.Path`.
    In some cases, this transform may insert curves into the path
    that began as line segments.
    Keywords: matplotlib.scale.SymmetricalLogTransform, transform_path
    """
    output_var = input_var.transform_path()

def matplotlib_scale_SymmetricalLogTransform_transform_path_affine(self: 'any') -> 'any_computed':
    r"""
    Apply the affine part of this transform to `.Path` *path*, returning a
    new `.Path`.
    ``transform_path(path)`` is equivalent to
    ``transform_path_affine(transform_path_non_affine(values))``.
    Keywords: matplotlib.scale.SymmetricalLogTransform, transform_path_affine
    """
    output_var = input_var.transform_path_affine()

def matplotlib_scale_SymmetricalLogTransform_transform_path_non_affine(self: 'any') -> 'any_computed':
    r"""
    Apply the non-affine part of this transform to `.Path` *path*,
    returning a new `.Path`.
    ``transform_path(path)`` is equivalent to
    ``transform_path_affine(transform_path_non_affine(values))``.
    Keywords: matplotlib.scale.SymmetricalLogTransform, transform_path_non_affine
    """
    output_var = input_var.transform_path_non_affine()

def matplotlib_scale_SymmetricalLogTransform_transform_point(self: 'any') -> 'any_computed':
    r"""
    Return a transformed point.
    This function is only kept for backcompatibility; the more general
    `.transform` method is capable of transforming both a list of points
    and a single point.
    Keywords: matplotlib.scale.SymmetricalLogTransform, transform_point
    """
    output_var = input_var.transform_point()

def matplotlib_scale_register_scale(scale_class: 'any') -> 'any_computed':
    r"""
    Register a new kind of scale.
    Parameters
    ----------
    scale_class : subclass of `ScaleBase`
    Keywords: matplotlib.scale, register_scale
    """
    output_var = matplotlib.scale.register_scale(input_var)

def matplotlib_scale_scale_factory(scale: 'any') -> 'any_computed':
    r"""
    Return a scale class by name.
    Parameters
    ----------
    scale : {'asinh', 'function', 'functionlog', 'linear', 'log', 'logit', 'symlog'}
    Keywords: matplotlib.scale, scale_factory
    """
    output_var = matplotlib.scale.scale_factory(input_var)

def matplotlib_colors_to_hex(c: 'any') -> 'any_computed':
    r"""
    Convert *c* to a hex color.
    Parameters
    ----------
    c : :ref:`color <colors_def>` or `numpy.ma.masked`
    Keywords: matplotlib.colors, to_hex
    """
    output_var = matplotlib.colors.to_hex(input_var)

def matplotlib_colors_to_rgb(c: 'any') -> 'any_computed':
    r"""
    Convert *c* to an RGB color, silently dropping the alpha channel.
    Keywords: matplotlib.colors, to_rgb
    """
    output_var = matplotlib.colors.to_rgb(input_var)

def matplotlib_colors_to_rgba(c: 'any') -> 'any_computed':
    r"""
    Convert *c* to an RGBA color.
    Parameters
    ----------
    c : Matplotlib color or ``np.ma.masked``
    Keywords: matplotlib.colors, to_rgba
    """
    output_var = matplotlib.colors.to_rgba(input_var)

def matplotlib_colors_to_rgba_array(c: 'any') -> 'any_computed':
    r"""
    Convert *c* to a (n, 4) array of RGBA colors.
    Parameters
    ----------
    c : Matplotlib color or array of colors
    Keywords: matplotlib.colors, to_rgba_array
    """
    output_var = matplotlib.colors.to_rgba_array(input_var)

def matplotlib_cm_get_cmap(name: 'any') -> 'any_computed':
    r"""
    [*Deprecated*] Get a colormap instance, defaulting to rc values if *name* is None.
    Parameters
    ----------
    name : `~matplotlib.colors.Colormap` or str or None, default: None
    Keywords: matplotlib.cm, get_cmap
    """
    output_var = matplotlib.cm.get_cmap(input_var)

def matplotlib_colorizer_Colorizer_autoscale(self: 'any') -> 'any_computed':
    r"""
    Autoscale the scalar limits on the norm instance using the
    current array
    Keywords: matplotlib.colorizer.Colorizer, autoscale
    """
    output_var = input_var.autoscale()

def matplotlib_colorizer_Colorizer_autoscale_None(self: 'any') -> 'any_computed':
    r"""
    Autoscale the scalar limits on the norm instance using the
    current array, changing only limits that are None
    Keywords: matplotlib.colorizer.Colorizer, autoscale_None
    """
    output_var = input_var.autoscale_None()

def matplotlib_colorizer_Colorizer_changed(self: 'any') -> 'any_computed':
    r"""
    Call this whenever the mappable is changed to notify all the
    callbackSM listeners to the 'changed' signal.
    Keywords: matplotlib.colorizer.Colorizer, changed
    """
    output_var = input_var.changed()

def matplotlib_colorizer_Colorizer_get_clim(self: 'any') -> 'any_computed':
    r"""
    Return the values (min, max) that are mapped to the colormap limits.
    Keywords: matplotlib.colorizer.Colorizer, get_clim
    """
    output_var = input_var.get_clim()

def matplotlib_colorizer_Colorizer_set_clim(self: 'any') -> 'any_computed':
    r"""
    Set the norm limits for image scaling.
    Parameters
    ----------
    vmin, vmax : float
    Keywords: matplotlib.colorizer.Colorizer, set_clim
    """
    output_var = input_var.set_clim()

def matplotlib_colorizer_Colorizer_to_rgba(self: 'any') -> 'any_computed':
    r"""
    Return a normalized RGBA array corresponding to *x*.
    In the normal case, *x* is a 1D or 2D sequence of scalars, and
    the corresponding `~numpy.ndarray` of RGBA values will be returned,
    based on the norm and colormap set for this Colorizer.
    Keywords: matplotlib.colorizer.Colorizer, to_rgba
    """
    output_var = input_var.to_rgba()

def matplotlib_colorizer_ColorizingArtist_add_callback(self: 'any') -> 'any_computed':
    r"""
    Add a callback function that will be called whenever one of the
    `.Artist`'s properties changes.
    Parameters
    ----------
    Keywords: matplotlib.colorizer.ColorizingArtist, add_callback
    """
    output_var = input_var.add_callback()

def matplotlib_colorizer_ColorizingArtist_autoscale(self: 'any') -> 'any_computed':
    r"""
    Autoscale the scalar limits on the norm instance using the
    current array
    Keywords: matplotlib.colorizer.ColorizingArtist, autoscale
    """
    output_var = input_var.autoscale()

def matplotlib_colorizer_ColorizingArtist_autoscale_None(self: 'any') -> 'any_computed':
    r"""
    Autoscale the scalar limits on the norm instance using the
    current array, changing only limits that are None
    Keywords: matplotlib.colorizer.ColorizingArtist, autoscale_None
    """
    output_var = input_var.autoscale_None()

def matplotlib_colorizer_ColorizingArtist_changed(self: 'any') -> 'any_computed':
    r"""
    Call this whenever the mappable is changed to notify all the
    callbackSM listeners to the 'changed' signal.
    Keywords: matplotlib.colorizer.ColorizingArtist, changed
    """
    output_var = input_var.changed()

def matplotlib_colorizer_ColorizingArtist_contains(self: 'any') -> 'any_computed':
    r"""
    Test whether the artist contains the mouse event.
    Parameters
    ----------
    mouseevent : `~matplotlib.backend_bases.MouseEvent`
    Keywords: matplotlib.colorizer.ColorizingArtist, contains
    """
    output_var = input_var.contains()

def matplotlib_colorizer_ColorizingArtist_convert_xunits(self: 'any') -> 'any_computed':
    r"""
    Convert *x* using the unit type of the xaxis.
    If the artist is not contained in an Axes or if the xaxis does not
    have units, *x* itself is returned.
    Keywords: matplotlib.colorizer.ColorizingArtist, convert_xunits
    """
    output_var = input_var.convert_xunits()

def matplotlib_colorizer_ColorizingArtist_convert_yunits(self: 'any') -> 'any_computed':
    r"""
    Convert *y* using the unit type of the yaxis.
    If the artist is not contained in an Axes or if the yaxis does not
    have units, *y* itself is returned.
    Keywords: matplotlib.colorizer.ColorizingArtist, convert_yunits
    """
    output_var = input_var.convert_yunits()

def matplotlib_colorizer_ColorizingArtist_draw(self: 'any') -> 'any_computed':
    r"""
    Draw the Artist (and its children) using the given renderer.
    This has no effect if the artist is not visible (`.Artist.get_visible`
    returns False).
    Keywords: matplotlib.colorizer.ColorizingArtist, draw
    """
    output_var = input_var.draw()

def matplotlib_colorizer_ColorizingArtist_findobj(self: 'any') -> 'any_computed':
    r"""
    Find artist objects.
    Recursively find all `.Artist` instances contained in the artist.
    Parameters
    Keywords: matplotlib.colorizer.ColorizingArtist, findobj
    """
    output_var = input_var.findobj()

def matplotlib_colorizer_ColorizingArtist_format_cursor_data(self: 'any') -> 'any_computed':
    r"""
    Return a string representation of *data*.
    .. note::
        This method is intended to be overridden by artist subclasses.
        As an end-user of Matplotlib you will most likely not call this
    Keywords: matplotlib.colorizer.ColorizingArtist, format_cursor_data
    """
    output_var = input_var.format_cursor_data()

def matplotlib_colorizer_ColorizingArtist_get_agg_filter(self: 'any') -> 'any_computed':
    r"""
    Return filter function to be used for agg filter.
    Keywords: matplotlib.colorizer.ColorizingArtist, get_agg_filter
    """
    output_var = input_var.get_agg_filter()

def matplotlib_colorizer_ColorizingArtist_get_alpha(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.colorizer.ColorizingArtist, get_alpha
    """
    output_var = input_var.get_alpha()

def matplotlib_colorizer_ColorizingArtist_get_animated(self: 'any') -> 'any_computed':
    r"""
    Return whether the artist is animated.
    Keywords: matplotlib.colorizer.ColorizingArtist, get_animated
    """
    output_var = input_var.get_animated()

def matplotlib_colorizer_ColorizingArtist_get_array(self: 'any') -> 'any_computed':
    r"""
    Return the array of values, that are mapped to colors.
    The base class `.ScalarMappable` does not make any assumptions on
    the dimensionality and shape of the array.
    Keywords: matplotlib.colorizer.ColorizingArtist, get_array
    """
    output_var = input_var.get_array()

def matplotlib_colorizer_ColorizingArtist_get_children(self: 'any') -> 'any_computed':
    r"""
    Return a list of the child `.Artist`\s of this `.Artist`.
    Keywords: matplotlib.colorizer.ColorizingArtist, get_children
    """
    output_var = input_var.get_children()

def matplotlib_colorizer_ColorizingArtist_get_clim(self: 'any') -> 'any_computed':
    r"""
    Return the values (min, max) that are mapped to the colormap limits.
    Keywords: matplotlib.colorizer.ColorizingArtist, get_clim
    """
    output_var = input_var.get_clim()

def matplotlib_colorizer_ColorizingArtist_get_clip_box(self: 'any') -> 'any_computed':
    r"""
    Return the clipbox.
    Keywords: matplotlib.colorizer.ColorizingArtist, get_clip_box
    """
    output_var = input_var.get_clip_box()

def matplotlib_colorizer_ColorizingArtist_get_clip_on(self: 'any') -> 'any_computed':
    r"""
    Return whether the artist uses clipping.
    Keywords: matplotlib.colorizer.ColorizingArtist, get_clip_on
    """
    output_var = input_var.get_clip_on()

def matplotlib_colorizer_ColorizingArtist_get_clip_path(self: 'any') -> 'any_computed':
    r"""
    Return the clip path.
    Keywords: matplotlib.colorizer.ColorizingArtist, get_clip_path
    """
    output_var = input_var.get_clip_path()

def matplotlib_colorizer_ColorizingArtist_get_cmap(self: 'any') -> 'any_computed':
    r"""
    Return the `.Colormap` instance.
    Keywords: matplotlib.colorizer.ColorizingArtist, get_cmap
    """
    output_var = input_var.get_cmap()

def matplotlib_colorizer_ColorizingArtist_get_cursor_data(self: 'any') -> 'any_computed':
    r"""
    Return the cursor data for a given event.
    .. note::
        This method is intended to be overridden by artist subclasses.
        As an end-user of Matplotlib you will most likely not call this
    Keywords: matplotlib.colorizer.ColorizingArtist, get_cursor_data
    """
    output_var = input_var.get_cursor_data()

def matplotlib_colorizer_ColorizingArtist_get_figure(self: 'any') -> 'any_computed':
    r"""
    Return the `.Figure` or `.SubFigure` instance the artist belongs to.
    Parameters
    ----------
    root : bool, default=False
    Keywords: matplotlib.colorizer.ColorizingArtist, get_figure
    """
    output_var = input_var.get_figure()

def matplotlib_colorizer_ColorizingArtist_get_gid(self: 'any') -> 'any_computed':
    r"""
    Return the group id.
    Keywords: matplotlib.colorizer.ColorizingArtist, get_gid
    """
    output_var = input_var.get_gid()

def matplotlib_colorizer_ColorizingArtist_get_in_layout(self: 'any') -> 'any_computed':
    r"""
    Return boolean flag, ``True`` if artist is included in layout
    calculations.
    E.g. :ref:`constrainedlayout_guide`,
    `.Figure.tight_layout()`, and
    Keywords: matplotlib.colorizer.ColorizingArtist, get_in_layout
    """
    output_var = input_var.get_in_layout()

def matplotlib_colorizer_ColorizingArtist_get_label(self: 'any') -> 'any_computed':
    r"""
    Return the label used for this artist in the legend.
    Keywords: matplotlib.colorizer.ColorizingArtist, get_label
    """
    output_var = input_var.get_label()

def matplotlib_colorizer_ColorizingArtist_get_mouseover(self: 'any') -> 'any_computed':
    r"""
    Return whether this artist is queried for custom context information
    when the mouse cursor moves over it.
    Keywords: matplotlib.colorizer.ColorizingArtist, get_mouseover
    """
    output_var = input_var.get_mouseover()

def matplotlib_colorizer_ColorizingArtist_get_path_effects(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.colorizer.ColorizingArtist, get_path_effects
    """
    output_var = input_var.get_path_effects()

def matplotlib_colorizer_ColorizingArtist_get_picker(self: 'any') -> 'any_computed':
    r"""
    Return the picking behavior of the artist.
    The possible values are described in `.Artist.set_picker`.
    See Also
    Keywords: matplotlib.colorizer.ColorizingArtist, get_picker
    """
    output_var = input_var.get_picker()

def matplotlib_colorizer_ColorizingArtist_get_rasterized(self: 'any') -> 'any_computed':
    r"""
    Return whether the artist is to be rasterized.
    Keywords: matplotlib.colorizer.ColorizingArtist, get_rasterized
    """
    output_var = input_var.get_rasterized()

def matplotlib_colorizer_ColorizingArtist_get_sketch_params(self: 'any') -> 'any_computed':
    r"""
    Return the sketch parameters for the artist.
    Returns
    -------
    tuple or None
    Keywords: matplotlib.colorizer.ColorizingArtist, get_sketch_params
    """
    output_var = input_var.get_sketch_params()

def matplotlib_colorizer_ColorizingArtist_get_snap(self: 'any') -> 'any_computed':
    r"""
    Return the snap setting.
    See `.set_snap` for details.
    Keywords: matplotlib.colorizer.ColorizingArtist, get_snap
    """
    output_var = input_var.get_snap()

def matplotlib_colorizer_ColorizingArtist_get_tightbbox(self: 'any') -> 'any_computed':
    r"""
    Like `.Artist.get_window_extent`, but includes any clipping.
    Parameters
    ----------
    renderer : `~matplotlib.backend_bases.RendererBase` subclass, optional
    Keywords: matplotlib.colorizer.ColorizingArtist, get_tightbbox
    """
    output_var = input_var.get_tightbbox()

def matplotlib_colorizer_ColorizingArtist_get_transform(self: 'any') -> 'any_computed':
    r"""
    Return the `.Transform` instance used by this artist.
    Keywords: matplotlib.colorizer.ColorizingArtist, get_transform
    """
    output_var = input_var.get_transform()

def matplotlib_colorizer_ColorizingArtist_get_transformed_clip_path_and_affine(self: 'any') -> 'any_computed':
    r"""
    Return the clip path with the non-affine part of its
    transformation applied, and the remaining affine part of its
    transformation.
    Keywords: matplotlib.colorizer.ColorizingArtist, get_transformed_clip_path_and_affine
    """
    output_var = input_var.get_transformed_clip_path_and_affine()

def matplotlib_colorizer_ColorizingArtist_get_url(self: 'any') -> 'any_computed':
    r"""
    Return the url.
    Keywords: matplotlib.colorizer.ColorizingArtist, get_url
    """
    output_var = input_var.get_url()

def matplotlib_colorizer_ColorizingArtist_get_visible(self: 'any') -> 'any_computed':
    r"""
    Return the visibility.
    Keywords: matplotlib.colorizer.ColorizingArtist, get_visible
    """
    output_var = input_var.get_visible()

def matplotlib_colorizer_ColorizingArtist_get_window_extent(self: 'any') -> 'any_computed':
    r"""
    Get the artist's bounding box in display space.
    The bounding box' width and height are nonnegative.
    Subclasses should override for inclusion in the bounding box
    Keywords: matplotlib.colorizer.ColorizingArtist, get_window_extent
    """
    output_var = input_var.get_window_extent()

def matplotlib_colorizer_ColorizingArtist_get_zorder(self: 'any') -> 'any_computed':
    r"""
    Return the artist's zorder.
    Keywords: matplotlib.colorizer.ColorizingArtist, get_zorder
    """
    output_var = input_var.get_zorder()

def matplotlib_colorizer_ColorizingArtist_have_units(self: 'any') -> 'any_computed':
    r"""
    Return whether units are set on any axis.
    Keywords: matplotlib.colorizer.ColorizingArtist, have_units
    """
    output_var = input_var.have_units()

def matplotlib_colorizer_ColorizingArtist_is_transform_set(self: 'any') -> 'any_computed':
    r"""
    Return whether the Artist has an explicitly set transform.
    This is *True* after `.set_transform` has been called.
    Keywords: matplotlib.colorizer.ColorizingArtist, is_transform_set
    """
    output_var = input_var.is_transform_set()

def matplotlib_colorizer_ColorizingArtist_pchanged(self: 'any') -> 'any_computed':
    r"""
    Call all of the registered callbacks.
    This function is triggered internally when a property is changed.
    See Also
    Keywords: matplotlib.colorizer.ColorizingArtist, pchanged
    """
    output_var = input_var.pchanged()

def matplotlib_colorizer_ColorizingArtist_pick(self: 'any') -> 'any_computed':
    r"""
    Process a pick event.
    Each child artist will fire a pick event if *mouseevent* is over
    the artist and the artist has picker set.
    Keywords: matplotlib.colorizer.ColorizingArtist, pick
    """
    output_var = input_var.pick()

def matplotlib_colorizer_ColorizingArtist_pickable(self: 'any') -> 'any_computed':
    r"""
    Return whether the artist is pickable.
    See Also
    --------
    .Artist.set_picker, .Artist.get_picker, .Artist.pick
    Keywords: matplotlib.colorizer.ColorizingArtist, pickable
    """
    output_var = input_var.pickable()

def matplotlib_colorizer_ColorizingArtist_properties(self: 'any') -> 'any_computed':
    r"""
    Return a dictionary of all the properties of the artist.
    Keywords: matplotlib.colorizer.ColorizingArtist, properties
    """
    output_var = input_var.properties()

def matplotlib_colorizer_ColorizingArtist_remove(self: 'any') -> 'any_computed':
    r"""
    Remove the artist from the figure if possible.
    The effect will not be visible until the figure is redrawn, e.g.,
    with `.FigureCanvasBase.draw_idle`.  Call `~.axes.Axes.relim` to
    update the Axes limits if desired.
    Keywords: matplotlib.colorizer.ColorizingArtist, remove
    """
    output_var = input_var.remove()

def matplotlib_colorizer_ColorizingArtist_remove_callback(self: 'any') -> 'any_computed':
    r"""
    Remove a callback based on its observer id.
    See Also
    --------
    add_callback
    Keywords: matplotlib.colorizer.ColorizingArtist, remove_callback
    """
    output_var = input_var.remove_callback()

def matplotlib_colorizer_ColorizingArtist_set(self: 'any') -> 'any_computed':
    r"""
    Set multiple properties at once.
    Supported properties are
    Properties:
    Keywords: matplotlib.colorizer.ColorizingArtist, set
    """
    output_var = input_var.set()

def matplotlib_colorizer_ColorizingArtist_set_agg_filter(self: 'any') -> 'any_computed':
    r"""
    Set the agg filter.
    Parameters
    ----------
    filter_func : callable
    Keywords: matplotlib.colorizer.ColorizingArtist, set_agg_filter
    """
    output_var = input_var.set_agg_filter()

def matplotlib_colorizer_ColorizingArtist_set_alpha(self: 'any') -> 'any_computed':
    r"""
    Set the alpha value used for blending - not supported on all backends.
    Parameters
    ----------
    alpha : float or None
    Keywords: matplotlib.colorizer.ColorizingArtist, set_alpha
    """
    output_var = input_var.set_alpha()

def matplotlib_colorizer_ColorizingArtist_set_animated(self: 'any') -> 'any_computed':
    r"""
    Set whether the artist is intended to be used in an animation.
    If True, the artist is excluded from regular drawing of the figure.
    You have to call `.Figure.draw_artist` / `.Axes.draw_artist`
    explicitly on the artist. This approach is used to speed up animations
    Keywords: matplotlib.colorizer.ColorizingArtist, set_animated
    """
    output_var = input_var.set_animated()

def matplotlib_colorizer_ColorizingArtist_set_array(self: 'any') -> 'any_computed':
    r"""
    Set the value array from array-like *A*.
    Parameters
    ----------
    A : array-like or None
    Keywords: matplotlib.colorizer.ColorizingArtist, set_array
    """
    output_var = input_var.set_array()

def matplotlib_colorizer_ColorizingArtist_set_clim(self: 'any') -> 'any_computed':
    r"""
    Set the norm limits for image scaling.
    Parameters
    ----------
    vmin, vmax : float
    Keywords: matplotlib.colorizer.ColorizingArtist, set_clim
    """
    output_var = input_var.set_clim()

def matplotlib_colorizer_ColorizingArtist_set_clip_box(self: 'any') -> 'any_computed':
    r"""
    Set the artist's clip `.Bbox`.
    Parameters
    ----------
    clipbox : `~matplotlib.transforms.BboxBase` or None
    Keywords: matplotlib.colorizer.ColorizingArtist, set_clip_box
    """
    output_var = input_var.set_clip_box()

def matplotlib_colorizer_ColorizingArtist_set_clip_on(self: 'any') -> 'any_computed':
    r"""
    Set whether the artist uses clipping.
    When False, artists will be visible outside the Axes which
    can lead to unexpected results.
    Keywords: matplotlib.colorizer.ColorizingArtist, set_clip_on
    """
    output_var = input_var.set_clip_on()

def matplotlib_colorizer_ColorizingArtist_set_clip_path(self: 'any') -> 'any_computed':
    r"""
    Set the artist's clip path.
    Parameters
    ----------
    path : `~matplotlib.patches.Patch` or `.Path` or `.TransformedPath` or None
    Keywords: matplotlib.colorizer.ColorizingArtist, set_clip_path
    """
    output_var = input_var.set_clip_path()

def matplotlib_colorizer_ColorizingArtist_set_cmap(self: 'any') -> 'any_computed':
    r"""
    Set the colormap for luminance data.
    Parameters
    ----------
    cmap : `.Colormap` or str or None
    Keywords: matplotlib.colorizer.ColorizingArtist, set_cmap
    """
    output_var = input_var.set_cmap()

def matplotlib_colorizer_ColorizingArtist_set_figure(self: 'any') -> 'any_computed':
    r"""
    Set the `.Figure` or `.SubFigure` instance the artist belongs to.
    Parameters
    ----------
    fig : `~matplotlib.figure.Figure` or `~matplotlib.figure.SubFigure`
    Keywords: matplotlib.colorizer.ColorizingArtist, set_figure
    """
    output_var = input_var.set_figure()

def matplotlib_colorizer_ColorizingArtist_set_gid(self: 'any') -> 'any_computed':
    r"""
    Set the (group) id for the artist.
    Parameters
    ----------
    gid : str
    Keywords: matplotlib.colorizer.ColorizingArtist, set_gid
    """
    output_var = input_var.set_gid()

def matplotlib_colorizer_ColorizingArtist_set_in_layout(self: 'any') -> 'any_computed':
    r"""
    Set if artist is to be included in layout calculations,
    E.g. :ref:`constrainedlayout_guide`,
    `.Figure.tight_layout()`, and
    ``fig.savefig(fname, bbox_inches='tight')``.
    Keywords: matplotlib.colorizer.ColorizingArtist, set_in_layout
    """
    output_var = input_var.set_in_layout()

def matplotlib_colorizer_ColorizingArtist_set_label(self: 'any') -> 'any_computed':
    r"""
    Set a label that will be displayed in the legend.
    Parameters
    ----------
    s : object
    Keywords: matplotlib.colorizer.ColorizingArtist, set_label
    """
    output_var = input_var.set_label()

def matplotlib_colorizer_ColorizingArtist_set_mouseover(self: 'any') -> 'any_computed':
    r"""
    Set whether this artist is queried for custom context information when
    the mouse cursor moves over it.
    Parameters
    ----------
    Keywords: matplotlib.colorizer.ColorizingArtist, set_mouseover
    """
    output_var = input_var.set_mouseover()

def matplotlib_colorizer_ColorizingArtist_set_norm(self: 'any') -> 'any_computed':
    r"""
    Set the normalization instance.
    Parameters
    ----------
    norm : `.Normalize` or str or None
    Keywords: matplotlib.colorizer.ColorizingArtist, set_norm
    """
    output_var = input_var.set_norm()

def matplotlib_colorizer_ColorizingArtist_set_path_effects(self: 'any') -> 'any_computed':
    r"""
    Set the path effects.
    Parameters
    ----------
    path_effects : list of `.AbstractPathEffect`
    Keywords: matplotlib.colorizer.ColorizingArtist, set_path_effects
    """
    output_var = input_var.set_path_effects()

def matplotlib_colorizer_ColorizingArtist_set_picker(self: 'any') -> 'any_computed':
    r"""
    Define the picking behavior of the artist.
    Parameters
    ----------
    picker : None or bool or float or callable
    Keywords: matplotlib.colorizer.ColorizingArtist, set_picker
    """
    output_var = input_var.set_picker()

def matplotlib_colorizer_ColorizingArtist_set_rasterized(self: 'any') -> 'any_computed':
    r"""
    Force rasterized (bitmap) drawing for vector graphics output.
    Rasterized drawing is not supported by all artists. If you try to
    enable this on an artist that does not support it, the command has no
    effect and a warning will be issued.
    Keywords: matplotlib.colorizer.ColorizingArtist, set_rasterized
    """
    output_var = input_var.set_rasterized()

def matplotlib_colorizer_ColorizingArtist_set_sketch_params(self: 'any') -> 'any_computed':
    r"""
    Set the sketch parameters.
    Parameters
    ----------
    scale : float, optional
    Keywords: matplotlib.colorizer.ColorizingArtist, set_sketch_params
    """
    output_var = input_var.set_sketch_params()

def matplotlib_colorizer_ColorizingArtist_set_snap(self: 'any') -> 'any_computed':
    r"""
    Set the snapping behavior.
    Snapping aligns positions with the pixel grid, which results in
    clearer images. For example, if a black line of 1px width was
    defined at a position in between two pixels, the resulting image
    Keywords: matplotlib.colorizer.ColorizingArtist, set_snap
    """
    output_var = input_var.set_snap()

def matplotlib_colorizer_ColorizingArtist_set_transform(self: 'any') -> 'any_computed':
    r"""
    Set the artist transform.
    Parameters
    ----------
    t : `~matplotlib.transforms.Transform`
    Keywords: matplotlib.colorizer.ColorizingArtist, set_transform
    """
    output_var = input_var.set_transform()

def matplotlib_colorizer_ColorizingArtist_set_url(self: 'any') -> 'any_computed':
    r"""
    Set the url for the artist.
    Parameters
    ----------
    url : str
    Keywords: matplotlib.colorizer.ColorizingArtist, set_url
    """
    output_var = input_var.set_url()

def matplotlib_colorizer_ColorizingArtist_set_visible(self: 'any') -> 'any_computed':
    r"""
    Set the artist's visibility.
    Parameters
    ----------
    b : bool
    Keywords: matplotlib.colorizer.ColorizingArtist, set_visible
    """
    output_var = input_var.set_visible()

def matplotlib_colorizer_ColorizingArtist_set_zorder(self: 'any') -> 'any_computed':
    r"""
    Set the zorder for the artist.  Artists with lower zorder
    values are drawn first.
    Parameters
    ----------
    Keywords: matplotlib.colorizer.ColorizingArtist, set_zorder
    """
    output_var = input_var.set_zorder()

def matplotlib_colorizer_ColorizingArtist_to_rgba(self: 'any') -> 'any_computed':
    r"""
    Return a normalized RGBA array corresponding to *x*.
    In the normal case, *x* is a 1D or 2D sequence of scalars, and
    the corresponding `~numpy.ndarray` of RGBA values will be returned,
    based on the norm and colormap set for this Colorizer.
    Keywords: matplotlib.colorizer.ColorizingArtist, to_rgba
    """
    output_var = input_var.to_rgba()

def matplotlib_colorizer_ColorizingArtist_update(self: 'any') -> 'any_computed':
    r"""
    Update this artist's properties from the dict *props*.
    Parameters
    ----------
    props : dict
    Keywords: matplotlib.colorizer.ColorizingArtist, update
    """
    output_var = input_var.update()

def matplotlib_colorizer_ColorizingArtist_update_from(self: 'any') -> 'any_computed':
    r"""
    Copy properties from *other* to *self*.
    Keywords: matplotlib.colorizer.ColorizingArtist, update_from
    """
    output_var = input_var.update_from()

def matplotlib_cycler(args: 'any') -> 'any_computed':
    r"""
    Create a `~cycler.Cycler` object much like :func:`cycler.cycler`,
    but includes input validation.
    Call signatures::
    Keywords: matplotlib, cycler
    """
    output_var = matplotlib.cycler(input_var)

def matplotlib_get_backend(auto_select: 'any') -> 'any_computed':
    r"""
    Return the name of the current backend.
    Parameters
    ----------
    auto_select : bool, default: True
    Keywords: matplotlib, get_backend
    """
    output_var = matplotlib.get_backend(input_var)

def matplotlib_interactive(b: 'any') -> 'any_computed':
    r"""
    Set whether to redraw after every plotting command (e.g. `.pyplot.xlabel`).
    Keywords: matplotlib, interactive
    """
    output_var = matplotlib.interactive(input_var)

def matplotlib_namedtuple(typename: 'any') -> 'any_computed':
    r"""
    Returns a new subclass of tuple with named fields.
    >>> Point = namedtuple('Point', ['x', 'y'])
    >>> Point.__doc__                   # docstring for the new class
    'Point(x, y)'
    Keywords: matplotlib, namedtuple
    """
    output_var = matplotlib.namedtuple(input_var)

def matplotlib_parse_version(version: 'str') -> 'Version_computed':
    r"""
    Parse the given version string.
    >>> parse('1.0.dev1')
    <Version('1.0.dev1')>
    Keywords: matplotlib, parse_version
    """
    output_var = matplotlib.parse_version(input_var)

def matplotlib_path_get_path_collection_extents(master_transform: 'any') -> 'any_computed':
    r"""
    Get bounding box of a `.PathCollection`\s internal objects.
    That is, given a sequence of `Path`\s, `.Transform`\s objects, and offsets, as found
    in a `.PathCollection`, return the bounding box that encapsulates all of them.
    Keywords: matplotlib.path, get_path_collection_extents
    """
    output_var = matplotlib.path.get_path_collection_extents(input_var)

def matplotlib_path_lru_cache(maxsize: 'any') -> 'any_computed':
    r"""
    Least-recently-used cache decorator.
    If *maxsize* is set to None, the LRU features are disabled and the cache
    can grow without bound.
    Keywords: matplotlib.path, lru_cache
    """
    output_var = matplotlib.path.lru_cache(input_var)

def matplotlib_path_simple_linear_interpolation(a: 'any') -> 'any_computed':
    r"""
    Resample an array with ``steps - 1`` points between original point pairs.
    Along each column of *a*, ``(steps - 1)`` points are introduced between
    each original values; the values are linearly interpolated.
    Keywords: matplotlib.path, simple_linear_interpolation
    """
    output_var = matplotlib.path.simple_linear_interpolation(input_var)

def matplotlib_rc(group: 'any') -> 'any_computed':
    r"""
    Set the current `.rcParams`.  *group* is the grouping for the rc, e.g.,
    for ``lines.linewidth`` the group is ``lines``, for
    ``axes.facecolor``, the group is ``axes``, and so on.  Group may
    also be a list or tuple of group names, e.g., (*xtick*, *ytick*).
    *kwargs* is a dictionary attribute name/value pairs, e.g.,::
    Keywords: matplotlib, rc
    """
    output_var = matplotlib.rc(input_var)

def matplotlib_rc_context(rc: 'any') -> 'any_computed':
    r"""
    Return a context manager for temporarily changing rcParams.
    The :rc:`backend` will not be reset by the context manager.
    rcParams changed both through the context manager invocation and
    Keywords: matplotlib, rc_context
    """
    output_var = matplotlib.rc_context(input_var)

def matplotlib_rc_file(fname: 'any') -> 'any_computed':
    r"""
    Update `.rcParams` from file.
    Style-blacklisted `.rcParams` (defined in
    ``matplotlib.style.core.STYLE_BLACKLIST``) are not updated.
    Keywords: matplotlib, rc_file
    """
    output_var = matplotlib.rc_file(input_var)

def matplotlib_rc_params(fail_on_error: 'any') -> 'any_computed':
    r"""
    Construct a `RcParams` instance from the default Matplotlib rc file.
    Keywords: matplotlib, rc_params
    """
    output_var = matplotlib.rc_params(input_var)

def matplotlib_rc_params_from_file(fname: 'any') -> 'any_computed':
    r"""
    Construct a `RcParams` from file *fname*.
    Parameters
    ----------
    fname : str or path-like
    Keywords: matplotlib, rc_params_from_file
    """
    output_var = matplotlib.rc_params_from_file(input_var)

def matplotlib_rcsetup_CapStyle_capitalize(self: 'any') -> 'any_computed':
    r"""
    Return a capitalized version of the string.
    More specifically, make the first character have upper case and the rest lower
    case.
    Keywords: matplotlib.rcsetup.CapStyle, capitalize
    """
    output_var = input_var.capitalize()

def matplotlib_rcsetup_CapStyle_casefold(self: 'any') -> 'any_computed':
    r"""
    Return a version of the string suitable for caseless comparisons.
    Keywords: matplotlib.rcsetup.CapStyle, casefold
    """
    output_var = input_var.casefold()

def matplotlib_rcsetup_CapStyle_center(self: 'any') -> 'any_computed':
    r"""
    Return a centered string of length width.
    Padding is done using the specified fill character (default is a space).
    Keywords: matplotlib.rcsetup.CapStyle, center
    """
    output_var = input_var.center()

def matplotlib_rcsetup_CapStyle_encode(self: 'any') -> 'any_computed':
    r"""
    Encode the string using the codec registered for encoding.
    encoding
      The encoding in which to encode the string.
    errors
    Keywords: matplotlib.rcsetup.CapStyle, encode
    """
    output_var = input_var.encode()

def matplotlib_rcsetup_CapStyle_expandtabs(self: 'any') -> 'any_computed':
    r"""
    Return a copy where all tab characters are expanded using spaces.
    If tabsize is not given, a tab size of 8 characters is assumed.
    Keywords: matplotlib.rcsetup.CapStyle, expandtabs
    """
    output_var = input_var.expandtabs()

def matplotlib_rcsetup_CapStyle_format(self: 'any') -> 'any_computed':
    r"""
    Return a formatted version of the string, using substitutions from args and kwargs.
    The substitutions are identified by braces ('{' and '}').
    Keywords: matplotlib.rcsetup.CapStyle, format
    """
    output_var = input_var.format()

def matplotlib_rcsetup_CapStyle_format_map(self: 'any') -> 'any_computed':
    r"""
    Return a formatted version of the string, using substitutions from mapping.
    The substitutions are identified by braces ('{' and '}').
    Keywords: matplotlib.rcsetup.CapStyle, format_map
    """
    output_var = input_var.format_map()

def matplotlib_rcsetup_CapStyle_isalnum(self: 'any') -> 'any_computed':
    r"""
    Return True if the string is an alpha-numeric string, False otherwise.
    A string is alpha-numeric if all characters in the string are alpha-numeric and
    there is at least one character in the string.
    Keywords: matplotlib.rcsetup.CapStyle, isalnum
    """
    output_var = input_var.isalnum()

def matplotlib_rcsetup_CapStyle_isalpha(self: 'any') -> 'any_computed':
    r"""
    Return True if the string is an alphabetic string, False otherwise.
    A string is alphabetic if all characters in the string are alphabetic and there
    is at least one character in the string.
    Keywords: matplotlib.rcsetup.CapStyle, isalpha
    """
    output_var = input_var.isalpha()

def matplotlib_rcsetup_CapStyle_isascii(self: 'any') -> 'any_computed':
    r"""
    Return True if all characters in the string are ASCII, False otherwise.
    ASCII characters have code points in the range U+0000-U+007F.
    Empty string is ASCII too.
    Keywords: matplotlib.rcsetup.CapStyle, isascii
    """
    output_var = input_var.isascii()

def matplotlib_rcsetup_CapStyle_isdecimal(self: 'any') -> 'any_computed':
    r"""
    Return True if the string is a decimal string, False otherwise.
    A string is a decimal string if all characters in the string are decimal and
    there is at least one character in the string.
    Keywords: matplotlib.rcsetup.CapStyle, isdecimal
    """
    output_var = input_var.isdecimal()

def matplotlib_rcsetup_CapStyle_isdigit(self: 'any') -> 'any_computed':
    r"""
    Return True if the string is a digit string, False otherwise.
    A string is a digit string if all characters in the string are digits and there
    is at least one character in the string.
    Keywords: matplotlib.rcsetup.CapStyle, isdigit
    """
    output_var = input_var.isdigit()

def matplotlib_rcsetup_CapStyle_isidentifier(self: 'any') -> 'any_computed':
    r"""
    Return True if the string is a valid Python identifier, False otherwise.
    Call keyword.iskeyword(s) to test whether string s is a reserved identifier,
    such as "def" or "class".
    Keywords: matplotlib.rcsetup.CapStyle, isidentifier
    """
    output_var = input_var.isidentifier()

def matplotlib_rcsetup_CapStyle_islower(self: 'any') -> 'any_computed':
    r"""
    Return True if the string is a lowercase string, False otherwise.
    A string is lowercase if all cased characters in the string are lowercase and
    there is at least one cased character in the string.
    Keywords: matplotlib.rcsetup.CapStyle, islower
    """
    output_var = input_var.islower()

def matplotlib_rcsetup_CapStyle_isnumeric(self: 'any') -> 'any_computed':
    r"""
    Return True if the string is a numeric string, False otherwise.
    A string is numeric if all characters in the string are numeric and there is at
    least one character in the string.
    Keywords: matplotlib.rcsetup.CapStyle, isnumeric
    """
    output_var = input_var.isnumeric()

def matplotlib_rcsetup_CapStyle_isprintable(self: 'any') -> 'any_computed':
    r"""
    Return True if the string is printable, False otherwise.
    A string is printable if all of its characters are considered printable in
    repr() or if it is empty.
    Keywords: matplotlib.rcsetup.CapStyle, isprintable
    """
    output_var = input_var.isprintable()

def matplotlib_rcsetup_CapStyle_isspace(self: 'any') -> 'any_computed':
    r"""
    Return True if the string is a whitespace string, False otherwise.
    A string is whitespace if all characters in the string are whitespace and there
    is at least one character in the string.
    Keywords: matplotlib.rcsetup.CapStyle, isspace
    """
    output_var = input_var.isspace()

def matplotlib_rcsetup_CapStyle_istitle(self: 'any') -> 'any_computed':
    r"""
    Return True if the string is a title-cased string, False otherwise.
    In a title-cased string, upper- and title-case characters may only
    follow uncased characters and lowercase characters only cased ones.
    Keywords: matplotlib.rcsetup.CapStyle, istitle
    """
    output_var = input_var.istitle()

def matplotlib_rcsetup_CapStyle_isupper(self: 'any') -> 'any_computed':
    r"""
    Return True if the string is an uppercase string, False otherwise.
    A string is uppercase if all cased characters in the string are uppercase and
    there is at least one cased character in the string.
    Keywords: matplotlib.rcsetup.CapStyle, isupper
    """
    output_var = input_var.isupper()

def matplotlib_rcsetup_CapStyle_join(self: 'any') -> 'any_computed':
    r"""
    Concatenate any number of strings.
    The string whose method is called is inserted in between each given string.
    The result is returned as a new string.
    Keywords: matplotlib.rcsetup.CapStyle, join
    """
    output_var = input_var.join()

def matplotlib_rcsetup_CapStyle_ljust(self: 'any') -> 'any_computed':
    r"""
    Return a left-justified string of length width.
    Padding is done using the specified fill character (default is a space).
    Keywords: matplotlib.rcsetup.CapStyle, ljust
    """
    output_var = input_var.ljust()

def matplotlib_rcsetup_CapStyle_lower(self: 'any') -> 'any_computed':
    r"""
    Return a copy of the string converted to lowercase.
    Keywords: matplotlib.rcsetup.CapStyle, lower
    """
    output_var = input_var.lower()

def matplotlib_rcsetup_CapStyle_lstrip(self: 'any') -> 'any_computed':
    r"""
    Return a copy of the string with leading whitespace removed.
    If chars is given and not None, remove characters in chars instead.
    Keywords: matplotlib.rcsetup.CapStyle, lstrip
    """
    output_var = input_var.lstrip()

def matplotlib_rcsetup_CapStyle_partition(self: 'any') -> 'any_computed':
    r"""
    Partition the string into three parts using the given separator.
    This will search for the separator in the string.  If the separator is found,
    returns a 3-tuple containing the part before the separator, the separator
    itself, and the part after it.
    Keywords: matplotlib.rcsetup.CapStyle, partition
    """
    output_var = input_var.partition()

def matplotlib_rcsetup_CapStyle_removeprefix(self: 'any') -> 'any_computed':
    r"""
    Return a str with the given prefix string removed if present.
    If the string starts with the prefix string, return string[len(prefix):].
    Otherwise, return a copy of the original string.
    Keywords: matplotlib.rcsetup.CapStyle, removeprefix
    """
    output_var = input_var.removeprefix()

def matplotlib_rcsetup_CapStyle_removesuffix(self: 'any') -> 'any_computed':
    r"""
    Return a str with the given suffix string removed if present.
    If the string ends with the suffix string and that suffix is not empty,
    return string[:-len(suffix)]. Otherwise, return a copy of the original
    string.
    Keywords: matplotlib.rcsetup.CapStyle, removesuffix
    """
    output_var = input_var.removesuffix()

def matplotlib_rcsetup_CapStyle_replace(self: 'any') -> 'any_computed':
    r"""
    Return a copy with all occurrences of substring old replaced by new.
      count
        Maximum number of occurrences to replace.
        -1 (the default value) means replace all occurrences.
    Keywords: matplotlib.rcsetup.CapStyle, replace
    """
    output_var = input_var.replace()

def matplotlib_rcsetup_CapStyle_rjust(self: 'any') -> 'any_computed':
    r"""
    Return a right-justified string of length width.
    Padding is done using the specified fill character (default is a space).
    Keywords: matplotlib.rcsetup.CapStyle, rjust
    """
    output_var = input_var.rjust()

def matplotlib_rcsetup_CapStyle_rpartition(self: 'any') -> 'any_computed':
    r"""
    Partition the string into three parts using the given separator.
    This will search for the separator in the string, starting at the end. If
    the separator is found, returns a 3-tuple containing the part before the
    separator, the separator itself, and the part after it.
    Keywords: matplotlib.rcsetup.CapStyle, rpartition
    """
    output_var = input_var.rpartition()

def matplotlib_rcsetup_CapStyle_rsplit(self: 'any') -> 'any_computed':
    r"""
    Return a list of the substrings in the string, using sep as the separator string.
      sep
        The separator used to split the string.
    Keywords: matplotlib.rcsetup.CapStyle, rsplit
    """
    output_var = input_var.rsplit()

def matplotlib_rcsetup_CapStyle_rstrip(self: 'any') -> 'any_computed':
    r"""
    Return a copy of the string with trailing whitespace removed.
    If chars is given and not None, remove characters in chars instead.
    Keywords: matplotlib.rcsetup.CapStyle, rstrip
    """
    output_var = input_var.rstrip()

def matplotlib_rcsetup_CapStyle_split(self: 'any') -> 'any_computed':
    r"""
    Return a list of the substrings in the string, using sep as the separator string.
      sep
        The separator used to split the string.
    Keywords: matplotlib.rcsetup.CapStyle, split
    """
    output_var = input_var.split()

def matplotlib_rcsetup_CapStyle_splitlines(self: 'any') -> 'any_computed':
    r"""
    Return a list of the lines in the string, breaking at line boundaries.
    Line breaks are not included in the resulting list unless keepends is given and
    true.
    Keywords: matplotlib.rcsetup.CapStyle, splitlines
    """
    output_var = input_var.splitlines()

def matplotlib_rcsetup_CapStyle_strip(self: 'any') -> 'any_computed':
    r"""
    Return a copy of the string with leading and trailing whitespace removed.
    If chars is given and not None, remove characters in chars instead.
    Keywords: matplotlib.rcsetup.CapStyle, strip
    """
    output_var = input_var.strip()

def matplotlib_rcsetup_CapStyle_swapcase(self: 'any') -> 'any_computed':
    r"""
    Convert uppercase characters to lowercase and lowercase characters to uppercase.
    Keywords: matplotlib.rcsetup.CapStyle, swapcase
    """
    output_var = input_var.swapcase()

def matplotlib_rcsetup_CapStyle_title(self: 'any') -> 'any_computed':
    r"""
    Return a version of the string where each word is titlecased.
    More specifically, words start with uppercased characters and all remaining
    cased characters have lower case.
    Keywords: matplotlib.rcsetup.CapStyle, title
    """
    output_var = input_var.title()

def matplotlib_rcsetup_CapStyle_translate(self: 'any') -> 'any_computed':
    r"""
    Replace each character in the string using the given translation table.
      table
        Translation table, which must be a mapping of Unicode ordinals to
        Unicode ordinals, strings, or None.
    Keywords: matplotlib.rcsetup.CapStyle, translate
    """
    output_var = input_var.translate()

def matplotlib_rcsetup_CapStyle_upper(self: 'any') -> 'any_computed':
    r"""
    Return a copy of the string converted to uppercase.
    Keywords: matplotlib.rcsetup.CapStyle, upper
    """
    output_var = input_var.upper()

def matplotlib_rcsetup_CapStyle_zfill(self: 'any') -> 'any_computed':
    r"""
    Pad a numeric string with zeros on the left, to fill a field of the given width.
    The string is never truncated.
    Keywords: matplotlib.rcsetup.CapStyle, zfill
    """
    output_var = input_var.zfill()

def matplotlib_rcsetup_JoinStyle_capitalize(self: 'any') -> 'any_computed':
    r"""
    Return a capitalized version of the string.
    More specifically, make the first character have upper case and the rest lower
    case.
    Keywords: matplotlib.rcsetup.JoinStyle, capitalize
    """
    output_var = input_var.capitalize()

def matplotlib_rcsetup_JoinStyle_casefold(self: 'any') -> 'any_computed':
    r"""
    Return a version of the string suitable for caseless comparisons.
    Keywords: matplotlib.rcsetup.JoinStyle, casefold
    """
    output_var = input_var.casefold()

def matplotlib_rcsetup_JoinStyle_center(self: 'any') -> 'any_computed':
    r"""
    Return a centered string of length width.
    Padding is done using the specified fill character (default is a space).
    Keywords: matplotlib.rcsetup.JoinStyle, center
    """
    output_var = input_var.center()

def matplotlib_rcsetup_JoinStyle_encode(self: 'any') -> 'any_computed':
    r"""
    Encode the string using the codec registered for encoding.
    encoding
      The encoding in which to encode the string.
    errors
    Keywords: matplotlib.rcsetup.JoinStyle, encode
    """
    output_var = input_var.encode()

def matplotlib_rcsetup_JoinStyle_expandtabs(self: 'any') -> 'any_computed':
    r"""
    Return a copy where all tab characters are expanded using spaces.
    If tabsize is not given, a tab size of 8 characters is assumed.
    Keywords: matplotlib.rcsetup.JoinStyle, expandtabs
    """
    output_var = input_var.expandtabs()

def matplotlib_rcsetup_JoinStyle_format(self: 'any') -> 'any_computed':
    r"""
    Return a formatted version of the string, using substitutions from args and kwargs.
    The substitutions are identified by braces ('{' and '}').
    Keywords: matplotlib.rcsetup.JoinStyle, format
    """
    output_var = input_var.format()

def matplotlib_rcsetup_JoinStyle_format_map(self: 'any') -> 'any_computed':
    r"""
    Return a formatted version of the string, using substitutions from mapping.
    The substitutions are identified by braces ('{' and '}').
    Keywords: matplotlib.rcsetup.JoinStyle, format_map
    """
    output_var = input_var.format_map()

def matplotlib_rcsetup_JoinStyle_isalnum(self: 'any') -> 'any_computed':
    r"""
    Return True if the string is an alpha-numeric string, False otherwise.
    A string is alpha-numeric if all characters in the string are alpha-numeric and
    there is at least one character in the string.
    Keywords: matplotlib.rcsetup.JoinStyle, isalnum
    """
    output_var = input_var.isalnum()

def matplotlib_rcsetup_JoinStyle_isalpha(self: 'any') -> 'any_computed':
    r"""
    Return True if the string is an alphabetic string, False otherwise.
    A string is alphabetic if all characters in the string are alphabetic and there
    is at least one character in the string.
    Keywords: matplotlib.rcsetup.JoinStyle, isalpha
    """
    output_var = input_var.isalpha()

def matplotlib_rcsetup_JoinStyle_isascii(self: 'any') -> 'any_computed':
    r"""
    Return True if all characters in the string are ASCII, False otherwise.
    ASCII characters have code points in the range U+0000-U+007F.
    Empty string is ASCII too.
    Keywords: matplotlib.rcsetup.JoinStyle, isascii
    """
    output_var = input_var.isascii()

def matplotlib_rcsetup_JoinStyle_isdecimal(self: 'any') -> 'any_computed':
    r"""
    Return True if the string is a decimal string, False otherwise.
    A string is a decimal string if all characters in the string are decimal and
    there is at least one character in the string.
    Keywords: matplotlib.rcsetup.JoinStyle, isdecimal
    """
    output_var = input_var.isdecimal()

def matplotlib_rcsetup_JoinStyle_isdigit(self: 'any') -> 'any_computed':
    r"""
    Return True if the string is a digit string, False otherwise.
    A string is a digit string if all characters in the string are digits and there
    is at least one character in the string.
    Keywords: matplotlib.rcsetup.JoinStyle, isdigit
    """
    output_var = input_var.isdigit()

def matplotlib_rcsetup_JoinStyle_isidentifier(self: 'any') -> 'any_computed':
    r"""
    Return True if the string is a valid Python identifier, False otherwise.
    Call keyword.iskeyword(s) to test whether string s is a reserved identifier,
    such as "def" or "class".
    Keywords: matplotlib.rcsetup.JoinStyle, isidentifier
    """
    output_var = input_var.isidentifier()

def matplotlib_rcsetup_JoinStyle_islower(self: 'any') -> 'any_computed':
    r"""
    Return True if the string is a lowercase string, False otherwise.
    A string is lowercase if all cased characters in the string are lowercase and
    there is at least one cased character in the string.
    Keywords: matplotlib.rcsetup.JoinStyle, islower
    """
    output_var = input_var.islower()

def matplotlib_rcsetup_JoinStyle_isnumeric(self: 'any') -> 'any_computed':
    r"""
    Return True if the string is a numeric string, False otherwise.
    A string is numeric if all characters in the string are numeric and there is at
    least one character in the string.
    Keywords: matplotlib.rcsetup.JoinStyle, isnumeric
    """
    output_var = input_var.isnumeric()

def matplotlib_rcsetup_JoinStyle_isprintable(self: 'any') -> 'any_computed':
    r"""
    Return True if the string is printable, False otherwise.
    A string is printable if all of its characters are considered printable in
    repr() or if it is empty.
    Keywords: matplotlib.rcsetup.JoinStyle, isprintable
    """
    output_var = input_var.isprintable()

def matplotlib_rcsetup_JoinStyle_isspace(self: 'any') -> 'any_computed':
    r"""
    Return True if the string is a whitespace string, False otherwise.
    A string is whitespace if all characters in the string are whitespace and there
    is at least one character in the string.
    Keywords: matplotlib.rcsetup.JoinStyle, isspace
    """
    output_var = input_var.isspace()

def matplotlib_rcsetup_JoinStyle_istitle(self: 'any') -> 'any_computed':
    r"""
    Return True if the string is a title-cased string, False otherwise.
    In a title-cased string, upper- and title-case characters may only
    follow uncased characters and lowercase characters only cased ones.
    Keywords: matplotlib.rcsetup.JoinStyle, istitle
    """
    output_var = input_var.istitle()

def matplotlib_rcsetup_JoinStyle_isupper(self: 'any') -> 'any_computed':
    r"""
    Return True if the string is an uppercase string, False otherwise.
    A string is uppercase if all cased characters in the string are uppercase and
    there is at least one cased character in the string.
    Keywords: matplotlib.rcsetup.JoinStyle, isupper
    """
    output_var = input_var.isupper()

def matplotlib_rcsetup_JoinStyle_join(self: 'any') -> 'any_computed':
    r"""
    Concatenate any number of strings.
    The string whose method is called is inserted in between each given string.
    The result is returned as a new string.
    Keywords: matplotlib.rcsetup.JoinStyle, join
    """
    output_var = input_var.join()

def matplotlib_rcsetup_JoinStyle_ljust(self: 'any') -> 'any_computed':
    r"""
    Return a left-justified string of length width.
    Padding is done using the specified fill character (default is a space).
    Keywords: matplotlib.rcsetup.JoinStyle, ljust
    """
    output_var = input_var.ljust()

def matplotlib_rcsetup_JoinStyle_lower(self: 'any') -> 'any_computed':
    r"""
    Return a copy of the string converted to lowercase.
    Keywords: matplotlib.rcsetup.JoinStyle, lower
    """
    output_var = input_var.lower()

def matplotlib_rcsetup_JoinStyle_lstrip(self: 'any') -> 'any_computed':
    r"""
    Return a copy of the string with leading whitespace removed.
    If chars is given and not None, remove characters in chars instead.
    Keywords: matplotlib.rcsetup.JoinStyle, lstrip
    """
    output_var = input_var.lstrip()

def matplotlib_rcsetup_JoinStyle_partition(self: 'any') -> 'any_computed':
    r"""
    Partition the string into three parts using the given separator.
    This will search for the separator in the string.  If the separator is found,
    returns a 3-tuple containing the part before the separator, the separator
    itself, and the part after it.
    Keywords: matplotlib.rcsetup.JoinStyle, partition
    """
    output_var = input_var.partition()

def matplotlib_rcsetup_JoinStyle_removeprefix(self: 'any') -> 'any_computed':
    r"""
    Return a str with the given prefix string removed if present.
    If the string starts with the prefix string, return string[len(prefix):].
    Otherwise, return a copy of the original string.
    Keywords: matplotlib.rcsetup.JoinStyle, removeprefix
    """
    output_var = input_var.removeprefix()

def matplotlib_rcsetup_JoinStyle_removesuffix(self: 'any') -> 'any_computed':
    r"""
    Return a str with the given suffix string removed if present.
    If the string ends with the suffix string and that suffix is not empty,
    return string[:-len(suffix)]. Otherwise, return a copy of the original
    string.
    Keywords: matplotlib.rcsetup.JoinStyle, removesuffix
    """
    output_var = input_var.removesuffix()

def matplotlib_rcsetup_JoinStyle_replace(self: 'any') -> 'any_computed':
    r"""
    Return a copy with all occurrences of substring old replaced by new.
      count
        Maximum number of occurrences to replace.
        -1 (the default value) means replace all occurrences.
    Keywords: matplotlib.rcsetup.JoinStyle, replace
    """
    output_var = input_var.replace()

def matplotlib_rcsetup_JoinStyle_rjust(self: 'any') -> 'any_computed':
    r"""
    Return a right-justified string of length width.
    Padding is done using the specified fill character (default is a space).
    Keywords: matplotlib.rcsetup.JoinStyle, rjust
    """
    output_var = input_var.rjust()

def matplotlib_rcsetup_JoinStyle_rpartition(self: 'any') -> 'any_computed':
    r"""
    Partition the string into three parts using the given separator.
    This will search for the separator in the string, starting at the end. If
    the separator is found, returns a 3-tuple containing the part before the
    separator, the separator itself, and the part after it.
    Keywords: matplotlib.rcsetup.JoinStyle, rpartition
    """
    output_var = input_var.rpartition()

def matplotlib_rcsetup_JoinStyle_rsplit(self: 'any') -> 'any_computed':
    r"""
    Return a list of the substrings in the string, using sep as the separator string.
      sep
        The separator used to split the string.
    Keywords: matplotlib.rcsetup.JoinStyle, rsplit
    """
    output_var = input_var.rsplit()

def matplotlib_rcsetup_JoinStyle_rstrip(self: 'any') -> 'any_computed':
    r"""
    Return a copy of the string with trailing whitespace removed.
    If chars is given and not None, remove characters in chars instead.
    Keywords: matplotlib.rcsetup.JoinStyle, rstrip
    """
    output_var = input_var.rstrip()

def matplotlib_rcsetup_JoinStyle_split(self: 'any') -> 'any_computed':
    r"""
    Return a list of the substrings in the string, using sep as the separator string.
      sep
        The separator used to split the string.
    Keywords: matplotlib.rcsetup.JoinStyle, split
    """
    output_var = input_var.split()

def matplotlib_rcsetup_JoinStyle_splitlines(self: 'any') -> 'any_computed':
    r"""
    Return a list of the lines in the string, breaking at line boundaries.
    Line breaks are not included in the resulting list unless keepends is given and
    true.
    Keywords: matplotlib.rcsetup.JoinStyle, splitlines
    """
    output_var = input_var.splitlines()

def matplotlib_rcsetup_JoinStyle_strip(self: 'any') -> 'any_computed':
    r"""
    Return a copy of the string with leading and trailing whitespace removed.
    If chars is given and not None, remove characters in chars instead.
    Keywords: matplotlib.rcsetup.JoinStyle, strip
    """
    output_var = input_var.strip()

def matplotlib_rcsetup_JoinStyle_swapcase(self: 'any') -> 'any_computed':
    r"""
    Convert uppercase characters to lowercase and lowercase characters to uppercase.
    Keywords: matplotlib.rcsetup.JoinStyle, swapcase
    """
    output_var = input_var.swapcase()

def matplotlib_rcsetup_JoinStyle_title(self: 'any') -> 'any_computed':
    r"""
    Return a version of the string where each word is titlecased.
    More specifically, words start with uppercased characters and all remaining
    cased characters have lower case.
    Keywords: matplotlib.rcsetup.JoinStyle, title
    """
    output_var = input_var.title()

def matplotlib_rcsetup_JoinStyle_translate(self: 'any') -> 'any_computed':
    r"""
    Replace each character in the string using the given translation table.
      table
        Translation table, which must be a mapping of Unicode ordinals to
        Unicode ordinals, strings, or None.
    Keywords: matplotlib.rcsetup.JoinStyle, translate
    """
    output_var = input_var.translate()

def matplotlib_rcsetup_JoinStyle_upper(self: 'any') -> 'any_computed':
    r"""
    Return a copy of the string converted to uppercase.
    Keywords: matplotlib.rcsetup.JoinStyle, upper
    """
    output_var = input_var.upper()

def matplotlib_rcsetup_JoinStyle_zfill(self: 'any') -> 'any_computed':
    r"""
    Pad a numeric string with zeros on the left, to fill a field of the given width.
    The string is never truncated.
    Keywords: matplotlib.rcsetup.JoinStyle, zfill
    """
    output_var = input_var.zfill()

def matplotlib_rcsetup_ccycler(args: 'any') -> 'any_computed':
    r"""
    Create a new `Cycler` object from a single positional argument,
    a pair of positional arguments, or the combination of keyword arguments.
    cycler(arg)
    cycler(label1=itr1[, label2=iter2[, ...]])
    Keywords: matplotlib.rcsetup, ccycler
    """
    output_var = matplotlib.rcsetup.ccycler(input_var)

def matplotlib_rcsetup_cycler(args: 'any') -> 'any_computed':
    r"""
    Create a `~cycler.Cycler` object much like :func:`cycler.cycler`,
    but includes input validation.
    Call signatures::
    Keywords: matplotlib.rcsetup, cycler
    """
    output_var = matplotlib.rcsetup.cycler(input_var)

def matplotlib_rcsetup_is_color_like(c: 'any') -> 'any_computed':
    r"""
    Return whether *c* can be interpreted as an RGB(A) color.
    Keywords: matplotlib.rcsetup, is_color_like
    """
    output_var = matplotlib.rcsetup.is_color_like(input_var)

def matplotlib_rcsetup_lru_cache(maxsize: 'any') -> 'any_computed':
    r"""
    Least-recently-used cache decorator.
    If *maxsize* is set to None, the LRU features are disabled and the cache
    can grow without bound.
    Keywords: matplotlib.rcsetup, lru_cache
    """
    output_var = matplotlib.rcsetup.lru_cache(input_var)

def matplotlib_rcsetup_parse_fontconfig_pattern(pattern: 'any') -> 'any_computed':
    r"""
    Parse a fontconfig *pattern* into a dict that can initialize a
    `.font_manager.FontProperties` object.
    Keywords: matplotlib.rcsetup, parse_fontconfig_pattern
    """
    output_var = matplotlib.rcsetup.parse_fontconfig_pattern(input_var)

def matplotlib_rcsetup_validate_any(s: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.rcsetup, validate_any
    """
    output_var = matplotlib.rcsetup.validate_any(input_var)

def matplotlib_rcsetup_validate_anylist(s: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.rcsetup, validate_anylist
    """
    output_var = matplotlib.rcsetup.validate_anylist(input_var)

def matplotlib_rcsetup_validate_aspect(s: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.rcsetup, validate_aspect
    """
    output_var = matplotlib.rcsetup.validate_aspect(input_var)

def matplotlib_rcsetup_validate_axisbelow(s: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.rcsetup, validate_axisbelow
    """
    output_var = matplotlib.rcsetup.validate_axisbelow(input_var)

def matplotlib_rcsetup_validate_backend(s: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.rcsetup, validate_backend
    """
    output_var = matplotlib.rcsetup.validate_backend(input_var)

def matplotlib_rcsetup_validate_bbox(s: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.rcsetup, validate_bbox
    """
    output_var = matplotlib.rcsetup.validate_bbox(input_var)

def matplotlib_rcsetup_validate_bool(b: 'any') -> 'any_computed':
    r"""
    Convert b to ``bool`` or raise.
    Keywords: matplotlib.rcsetup, validate_bool
    """
    output_var = matplotlib.rcsetup.validate_bool(input_var)

def matplotlib_rcsetup_validate_color(s: 'any') -> 'any_computed':
    r"""
    Return a valid color arg.
    Keywords: matplotlib.rcsetup, validate_color
    """
    output_var = matplotlib.rcsetup.validate_color(input_var)

def matplotlib_rcsetup_validate_color_for_prop_cycle(s: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.rcsetup, validate_color_for_prop_cycle
    """
    output_var = matplotlib.rcsetup.validate_color_for_prop_cycle(input_var)

def matplotlib_rcsetup_validate_color_or_auto(s: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.rcsetup, validate_color_or_auto
    """
    output_var = matplotlib.rcsetup.validate_color_or_auto(input_var)

def matplotlib_rcsetup_validate_color_or_inherit(s: 'any') -> 'any_computed':
    r"""
    Return a valid color arg.
    Keywords: matplotlib.rcsetup, validate_color_or_inherit
    """
    output_var = matplotlib.rcsetup.validate_color_or_inherit(input_var)

def matplotlib_rcsetup_validate_colorlist(s: 'any') -> 'any_computed':
    r"""
    return a list of colorspecs
    Keywords: matplotlib.rcsetup, validate_colorlist
    """
    output_var = matplotlib.rcsetup.validate_colorlist(input_var)

def matplotlib_rcsetup_validate_cycler(s: 'any') -> 'any_computed':
    r"""
    Return a Cycler object from a string repr or the object itself.
    Keywords: matplotlib.rcsetup, validate_cycler
    """
    output_var = matplotlib.rcsetup.validate_cycler(input_var)

def matplotlib_rcsetup_validate_dashlist(s: 'any') -> 'any_computed':
    r"""
    return a list of floats
    Keywords: matplotlib.rcsetup, validate_dashlist
    """
    output_var = matplotlib.rcsetup.validate_dashlist(input_var)

def matplotlib_rcsetup_validate_dpi(s: 'any') -> 'any_computed':
    r"""
    Confirm s is string 'figure' or convert s to float or raise.
    Keywords: matplotlib.rcsetup, validate_dpi
    """
    output_var = matplotlib.rcsetup.validate_dpi(input_var)

def matplotlib_rcsetup_validate_fillstylelist(s: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.rcsetup, validate_fillstylelist
    """
    output_var = matplotlib.rcsetup.validate_fillstylelist(input_var)

def matplotlib_rcsetup_validate_float(s: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.rcsetup, validate_float
    """
    output_var = matplotlib.rcsetup.validate_float(input_var)

def matplotlib_rcsetup_validate_float_or_None(s: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.rcsetup, validate_float_or_None
    """
    output_var = matplotlib.rcsetup.validate_float_or_None(input_var)

def matplotlib_rcsetup_validate_floatlist(s: 'any') -> 'any_computed':
    r"""
    return a list of floats
    Keywords: matplotlib.rcsetup, validate_floatlist
    """
    output_var = matplotlib.rcsetup.validate_floatlist(input_var)

def matplotlib_rcsetup_validate_font_properties(s: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.rcsetup, validate_font_properties
    """
    output_var = matplotlib.rcsetup.validate_font_properties(input_var)

def matplotlib_rcsetup_validate_fontsize(s: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.rcsetup, validate_fontsize
    """
    output_var = matplotlib.rcsetup.validate_fontsize(input_var)

def matplotlib_rcsetup_validate_fontsize_None(s: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.rcsetup, validate_fontsize_None
    """
    output_var = matplotlib.rcsetup.validate_fontsize_None(input_var)

def matplotlib_rcsetup_validate_fontsizelist(s: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.rcsetup, validate_fontsizelist
    """
    output_var = matplotlib.rcsetup.validate_fontsizelist(input_var)

def matplotlib_rcsetup_validate_fontstretch(s: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.rcsetup, validate_fontstretch
    """
    output_var = matplotlib.rcsetup.validate_fontstretch(input_var)

def matplotlib_rcsetup_validate_fonttype(s: 'any') -> 'any_computed':
    r"""
    Confirm that this is a Postscript or PDF font type that we know how to
    convert to.
    Keywords: matplotlib.rcsetup, validate_fonttype
    """
    output_var = matplotlib.rcsetup.validate_fonttype(input_var)

def matplotlib_rcsetup_validate_fontweight(s: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.rcsetup, validate_fontweight
    """
    output_var = matplotlib.rcsetup.validate_fontweight(input_var)

def matplotlib_rcsetup_validate_hatch(s: 'any') -> 'any_computed':
    r"""
    Validate a hatch pattern.
    A hatch pattern string can have any sequence of the following
    characters: ``\ / | - + * . x o O``.
    Keywords: matplotlib.rcsetup, validate_hatch
    """
    output_var = matplotlib.rcsetup.validate_hatch(input_var)

def matplotlib_rcsetup_validate_hatchlist(s: 'any') -> 'any_computed':
    r"""
    Validate a hatch pattern.
    A hatch pattern string can have any sequence of the following
    characters: ``\ / | - + * . x o O``.
    Keywords: matplotlib.rcsetup, validate_hatchlist
    """
    output_var = matplotlib.rcsetup.validate_hatchlist(input_var)

def matplotlib_rcsetup_validate_hist_bins(s: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.rcsetup, validate_hist_bins
    """
    output_var = matplotlib.rcsetup.validate_hist_bins(input_var)

def matplotlib_rcsetup_validate_int(s: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.rcsetup, validate_int
    """
    output_var = matplotlib.rcsetup.validate_int(input_var)

def matplotlib_rcsetup_validate_int_or_None(s: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.rcsetup, validate_int_or_None
    """
    output_var = matplotlib.rcsetup.validate_int_or_None(input_var)

def matplotlib_rcsetup_validate_markevery(s: 'any') -> 'any_computed':
    r"""
    Validate the markevery property of a Line2D object.
    Parameters
    ----------
    s : None, int, (int, int), slice, float, (float, float), or list[int]
    Keywords: matplotlib.rcsetup, validate_markevery
    """
    output_var = matplotlib.rcsetup.validate_markevery(input_var)

def matplotlib_rcsetup_validate_markeverylist(s: 'any') -> 'any_computed':
    r"""
    Validate the markevery property of a Line2D object.
    Parameters
    ----------
    s : None, int, (int, int), slice, float, (float, float), or list[int]
    Keywords: matplotlib.rcsetup, validate_markeverylist
    """
    output_var = matplotlib.rcsetup.validate_markeverylist(input_var)

def matplotlib_rcsetup_validate_ps_distiller(s: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.rcsetup, validate_ps_distiller
    """
    output_var = matplotlib.rcsetup.validate_ps_distiller(input_var)

def matplotlib_rcsetup_validate_sketch(s: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.rcsetup, validate_sketch
    """
    output_var = matplotlib.rcsetup.validate_sketch(input_var)

def matplotlib_rcsetup_validate_string(s: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.rcsetup, validate_string
    """
    output_var = matplotlib.rcsetup.validate_string(input_var)

def matplotlib_rcsetup_validate_string_or_None(s: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.rcsetup, validate_string_or_None
    """
    output_var = matplotlib.rcsetup.validate_string_or_None(input_var)

def matplotlib_rcsetup_validate_stringlist(s: 'any') -> 'any_computed':
    r"""
    return a list of strings
    Keywords: matplotlib.rcsetup, validate_stringlist
    """
    output_var = matplotlib.rcsetup.validate_stringlist(input_var)

def matplotlib_rcsetup_validate_whiskers(s: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.rcsetup, validate_whiskers
    """
    output_var = matplotlib.rcsetup.validate_whiskers(input_var)

def matplotlib_sanitize_sequence(data: 'any') -> 'any_computed':
    r"""
    [*Deprecated*] 
    Notes
    -----
    .. deprecated:: 3.10
    Keywords: matplotlib, sanitize_sequence
    """
    output_var = matplotlib.sanitize_sequence(input_var)

def matplotlib_set_loglevel(level: 'any') -> 'any_computed':
    r"""
    Configure Matplotlib's logging levels.
    Matplotlib uses the standard library `logging` framework under the root
    logger 'matplotlib'.  This is a helper function to:
    Keywords: matplotlib, set_loglevel
    """
    output_var = matplotlib.set_loglevel(input_var)

def matplotlib_ticker_EngFormatter_create_dummy_axis(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.ticker.EngFormatter, create_dummy_axis
    """
    output_var = input_var.create_dummy_axis()

def matplotlib_ticker_EngFormatter_fix_minus(s: 'any') -> 'any_computed':
    r"""
    Some classes may want to replace a hyphen for minus with the proper
    Unicode symbol (U+2212) for typographical correctness.  This is a
    helper method to perform such a replacement when it is enabled via
    :rc:`axes.unicode_minus`.
    Keywords: matplotlib.ticker.EngFormatter, fix_minus
    """
    output_var = input_var.fix_minus()

def matplotlib_ticker_EngFormatter_format_data(self: 'any') -> 'any_computed':
    r"""
    Format a number in engineering notation, appending a letter
    representing the power of 1000 of the original number.
    Some examples:
    >>> format_data(0)        # for self.places = 0
    Keywords: matplotlib.ticker.EngFormatter, format_data
    """
    output_var = input_var.format_data()

def matplotlib_ticker_EngFormatter_format_data_short(self: 'any') -> 'any_computed':
    r"""
    Return a short string version of the tick value.
    Defaults to the position-independent long value.
    Keywords: matplotlib.ticker.EngFormatter, format_data_short
    """
    output_var = input_var.format_data_short()

def matplotlib_ticker_EngFormatter_format_eng(self: 'any') -> 'any_computed':
    r"""
    Alias to EngFormatter.format_data
    Keywords: matplotlib.ticker.EngFormatter, format_eng
    """
    output_var = input_var.format_eng()

def matplotlib_ticker_EngFormatter_format_ticks(self: 'any') -> 'any_computed':
    r"""
    Return the tick labels for all the ticks at once.
    Keywords: matplotlib.ticker.EngFormatter, format_ticks
    """
    output_var = input_var.format_ticks()

def matplotlib_ticker_EngFormatter_get_offset(self: 'any') -> 'any_computed':
    r"""
    Return scientific notation, plus offset.
    Keywords: matplotlib.ticker.EngFormatter, get_offset
    """
    output_var = input_var.get_offset()

def matplotlib_ticker_EngFormatter_get_useLocale(self: 'any') -> 'any_computed':
    r"""
    Return whether locale settings are used for formatting.
    See Also
    --------
    ScalarFormatter.set_useLocale
    Keywords: matplotlib.ticker.EngFormatter, get_useLocale
    """
    output_var = input_var.get_useLocale()

def matplotlib_ticker_EngFormatter_get_useMathText(self: 'any') -> 'any_computed':
    r"""
    Return whether to use fancy math formatting.
    See Also
    --------
    ScalarFormatter.set_useMathText
    Keywords: matplotlib.ticker.EngFormatter, get_useMathText
    """
    output_var = input_var.get_useMathText()

def matplotlib_ticker_EngFormatter_get_useOffset(self: 'any') -> 'any_computed':
    r"""
    Return whether automatic mode for offset notation is active.
    This returns True if ``set_useOffset(True)``; it returns False if an
    explicit offset was set, e.g. ``set_useOffset(1000)``.
    Keywords: matplotlib.ticker.EngFormatter, get_useOffset
    """
    output_var = input_var.get_useOffset()

def matplotlib_ticker_EngFormatter_get_usetex(self: 'any') -> 'any_computed':
    r"""
    Return whether TeX's math mode is enabled for rendering.
    Keywords: matplotlib.ticker.EngFormatter, get_usetex
    """
    output_var = input_var.get_usetex()

def matplotlib_ticker_EngFormatter_set_axis(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.ticker.EngFormatter, set_axis
    """
    output_var = input_var.set_axis()

def matplotlib_ticker_EngFormatter_set_locs(self: 'any') -> 'any_computed':
    r"""
    Set the locations of the ticks.
    This method is called before computing the tick labels because some
    formatters need to know all tick locations to do so.
    Keywords: matplotlib.ticker.EngFormatter, set_locs
    """
    output_var = input_var.set_locs()

def matplotlib_ticker_EngFormatter_set_powerlimits(self: 'any') -> 'any_computed':
    r"""
    Set size thresholds for scientific notation.
    Parameters
    ----------
    lims : (int, int)
    Keywords: matplotlib.ticker.EngFormatter, set_powerlimits
    """
    output_var = input_var.set_powerlimits()

def matplotlib_ticker_EngFormatter_set_scientific(self: 'any') -> 'any_computed':
    r"""
    Turn scientific notation on or off.
    See Also
    --------
    ScalarFormatter.set_powerlimits
    Keywords: matplotlib.ticker.EngFormatter, set_scientific
    """
    output_var = input_var.set_scientific()

def matplotlib_ticker_EngFormatter_set_useLocale(self: 'any') -> 'any_computed':
    r"""
    Set whether to use locale settings for decimal sign and positive sign.
    Parameters
    ----------
    val : bool or None
    Keywords: matplotlib.ticker.EngFormatter, set_useLocale
    """
    output_var = input_var.set_useLocale()

def matplotlib_ticker_EngFormatter_set_useMathText(self: 'any') -> 'any_computed':
    r"""
    Set whether to use fancy math formatting.
    If active, scientific notation is formatted as :math:`1.2 \times 10^3`.
    Parameters
    Keywords: matplotlib.ticker.EngFormatter, set_useMathText
    """
    output_var = input_var.set_useMathText()

def matplotlib_ticker_EngFormatter_set_useOffset(self: 'any') -> 'any_computed':
    r"""
    Set whether to use offset notation.
    When formatting a set numbers whose value is large compared to their
    range, the formatter can separate an additive constant. This can
    shorten the formatted numbers so that they are less likely to overlap
    Keywords: matplotlib.ticker.EngFormatter, set_useOffset
    """
    output_var = input_var.set_useOffset()

def matplotlib_ticker_EngFormatter_set_usetex(self: 'any') -> 'any_computed':
    r"""
    Set whether to use TeX's math mode for rendering numbers in the formatter.
    Keywords: matplotlib.ticker.EngFormatter, set_usetex
    """
    output_var = input_var.set_usetex()

def matplotlib_ticker_FixedFormatter_create_dummy_axis(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.ticker.FixedFormatter, create_dummy_axis
    """
    output_var = input_var.create_dummy_axis()

def matplotlib_ticker_FixedFormatter_fix_minus(s: 'any') -> 'any_computed':
    r"""
    Some classes may want to replace a hyphen for minus with the proper
    Unicode symbol (U+2212) for typographical correctness.  This is a
    helper method to perform such a replacement when it is enabled via
    :rc:`axes.unicode_minus`.
    Keywords: matplotlib.ticker.FixedFormatter, fix_minus
    """
    output_var = input_var.fix_minus()

def matplotlib_ticker_FixedFormatter_format_data(self: 'any') -> 'any_computed':
    r"""
    Return the full string representation of the value with the
    position unspecified.
    Keywords: matplotlib.ticker.FixedFormatter, format_data
    """
    output_var = input_var.format_data()

def matplotlib_ticker_FixedFormatter_format_data_short(self: 'any') -> 'any_computed':
    r"""
    Return a short string version of the tick value.
    Defaults to the position-independent long value.
    Keywords: matplotlib.ticker.FixedFormatter, format_data_short
    """
    output_var = input_var.format_data_short()

def matplotlib_ticker_FixedFormatter_format_ticks(self: 'any') -> 'any_computed':
    r"""
    Return the tick labels for all the ticks at once.
    Keywords: matplotlib.ticker.FixedFormatter, format_ticks
    """
    output_var = input_var.format_ticks()

def matplotlib_ticker_FixedFormatter_get_offset(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.ticker.FixedFormatter, get_offset
    """
    output_var = input_var.get_offset()

def matplotlib_ticker_FixedFormatter_set_axis(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.ticker.FixedFormatter, set_axis
    """
    output_var = input_var.set_axis()

def matplotlib_ticker_FixedFormatter_set_locs(self: 'any') -> 'any_computed':
    r"""
    Set the locations of the ticks.
    This method is called before computing the tick labels because some
    formatters need to know all tick locations to do so.
    Keywords: matplotlib.ticker.FixedFormatter, set_locs
    """
    output_var = input_var.set_locs()

def matplotlib_ticker_FixedFormatter_set_offset_string(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.ticker.FixedFormatter, set_offset_string
    """
    output_var = input_var.set_offset_string()

def matplotlib_ticker_FixedLocator_create_dummy_axis(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.ticker.FixedLocator, create_dummy_axis
    """
    output_var = input_var.create_dummy_axis()

def matplotlib_ticker_FixedLocator_nonsingular(self: 'any') -> 'any_computed':
    r"""
    Adjust a range as needed to avoid singularities.
    This method gets called during autoscaling, with ``(v0, v1)`` set to
    the data limits on the Axes if the Axes contains any data, or
    ``(-inf, +inf)`` if not.
    Keywords: matplotlib.ticker.FixedLocator, nonsingular
    """
    output_var = input_var.nonsingular()

def matplotlib_ticker_FixedLocator_raise_if_exceeds(self: 'any') -> 'any_computed':
    r"""
    Log at WARNING level if *locs* is longer than `Locator.MAXTICKS`.
    This is intended to be called immediately before returning *locs* from
    ``__call__`` to inform users in case their Locator returns a huge
    number of ticks, causing Matplotlib to run out of memory.
    Keywords: matplotlib.ticker.FixedLocator, raise_if_exceeds
    """
    output_var = input_var.raise_if_exceeds()

def matplotlib_ticker_FixedLocator_set_axis(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.ticker.FixedLocator, set_axis
    """
    output_var = input_var.set_axis()

def matplotlib_ticker_FixedLocator_set_params(self: 'any') -> 'any_computed':
    r"""
    Set parameters within this locator.
    Keywords: matplotlib.ticker.FixedLocator, set_params
    """
    output_var = input_var.set_params()

def matplotlib_ticker_FixedLocator_tick_values(self: 'any') -> 'any_computed':
    r"""
    Return the locations of the ticks.
    .. note::
        Because the values are fixed, vmin and vmax are not used in this
    Keywords: matplotlib.ticker.FixedLocator, tick_values
    """
    output_var = input_var.tick_values()

def matplotlib_ticker_FixedLocator_view_limits(self: 'any') -> 'any_computed':
    r"""
    Select a scale for the range from vmin to vmax.
    Subclasses should override this method to change locator behaviour.
    Keywords: matplotlib.ticker.FixedLocator, view_limits
    """
    output_var = input_var.view_limits()

def matplotlib_ticker_FormatStrFormatter_create_dummy_axis(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.ticker.FormatStrFormatter, create_dummy_axis
    """
    output_var = input_var.create_dummy_axis()

def matplotlib_ticker_FormatStrFormatter_fix_minus(s: 'any') -> 'any_computed':
    r"""
    Some classes may want to replace a hyphen for minus with the proper
    Unicode symbol (U+2212) for typographical correctness.  This is a
    helper method to perform such a replacement when it is enabled via
    :rc:`axes.unicode_minus`.
    Keywords: matplotlib.ticker.FormatStrFormatter, fix_minus
    """
    output_var = input_var.fix_minus()

def matplotlib_ticker_FormatStrFormatter_format_data(self: 'any') -> 'any_computed':
    r"""
    Return the full string representation of the value with the
    position unspecified.
    Keywords: matplotlib.ticker.FormatStrFormatter, format_data
    """
    output_var = input_var.format_data()

def matplotlib_ticker_FormatStrFormatter_format_data_short(self: 'any') -> 'any_computed':
    r"""
    Return a short string version of the tick value.
    Defaults to the position-independent long value.
    Keywords: matplotlib.ticker.FormatStrFormatter, format_data_short
    """
    output_var = input_var.format_data_short()

def matplotlib_ticker_FormatStrFormatter_format_ticks(self: 'any') -> 'any_computed':
    r"""
    Return the tick labels for all the ticks at once.
    Keywords: matplotlib.ticker.FormatStrFormatter, format_ticks
    """
    output_var = input_var.format_ticks()

def matplotlib_ticker_FormatStrFormatter_get_offset(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.ticker.FormatStrFormatter, get_offset
    """
    output_var = input_var.get_offset()

def matplotlib_ticker_FormatStrFormatter_set_axis(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.ticker.FormatStrFormatter, set_axis
    """
    output_var = input_var.set_axis()

def matplotlib_ticker_FormatStrFormatter_set_locs(self: 'any') -> 'any_computed':
    r"""
    Set the locations of the ticks.
    This method is called before computing the tick labels because some
    formatters need to know all tick locations to do so.
    Keywords: matplotlib.ticker.FormatStrFormatter, set_locs
    """
    output_var = input_var.set_locs()

def matplotlib_ticker_Formatter_create_dummy_axis(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.ticker.Formatter, create_dummy_axis
    """
    output_var = input_var.create_dummy_axis()

def matplotlib_ticker_Formatter_fix_minus(s: 'any') -> 'any_computed':
    r"""
    Some classes may want to replace a hyphen for minus with the proper
    Unicode symbol (U+2212) for typographical correctness.  This is a
    helper method to perform such a replacement when it is enabled via
    :rc:`axes.unicode_minus`.
    Keywords: matplotlib.ticker.Formatter, fix_minus
    """
    output_var = input_var.fix_minus()

def matplotlib_ticker_Formatter_format_data(self: 'any') -> 'any_computed':
    r"""
    Return the full string representation of the value with the
    position unspecified.
    Keywords: matplotlib.ticker.Formatter, format_data
    """
    output_var = input_var.format_data()

def matplotlib_ticker_Formatter_format_data_short(self: 'any') -> 'any_computed':
    r"""
    Return a short string version of the tick value.
    Defaults to the position-independent long value.
    Keywords: matplotlib.ticker.Formatter, format_data_short
    """
    output_var = input_var.format_data_short()

def matplotlib_ticker_Formatter_format_ticks(self: 'any') -> 'any_computed':
    r"""
    Return the tick labels for all the ticks at once.
    Keywords: matplotlib.ticker.Formatter, format_ticks
    """
    output_var = input_var.format_ticks()

def matplotlib_ticker_Formatter_get_offset(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.ticker.Formatter, get_offset
    """
    output_var = input_var.get_offset()

def matplotlib_ticker_Formatter_set_axis(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.ticker.Formatter, set_axis
    """
    output_var = input_var.set_axis()

def matplotlib_ticker_Formatter_set_locs(self: 'any') -> 'any_computed':
    r"""
    Set the locations of the ticks.
    This method is called before computing the tick labels because some
    formatters need to know all tick locations to do so.
    Keywords: matplotlib.ticker.Formatter, set_locs
    """
    output_var = input_var.set_locs()

def matplotlib_ticker_FuncFormatter_create_dummy_axis(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.ticker.FuncFormatter, create_dummy_axis
    """
    output_var = input_var.create_dummy_axis()

def matplotlib_ticker_FuncFormatter_fix_minus(s: 'any') -> 'any_computed':
    r"""
    Some classes may want to replace a hyphen for minus with the proper
    Unicode symbol (U+2212) for typographical correctness.  This is a
    helper method to perform such a replacement when it is enabled via
    :rc:`axes.unicode_minus`.
    Keywords: matplotlib.ticker.FuncFormatter, fix_minus
    """
    output_var = input_var.fix_minus()

def matplotlib_ticker_FuncFormatter_format_data(self: 'any') -> 'any_computed':
    r"""
    Return the full string representation of the value with the
    position unspecified.
    Keywords: matplotlib.ticker.FuncFormatter, format_data
    """
    output_var = input_var.format_data()

def matplotlib_ticker_FuncFormatter_format_data_short(self: 'any') -> 'any_computed':
    r"""
    Return a short string version of the tick value.
    Defaults to the position-independent long value.
    Keywords: matplotlib.ticker.FuncFormatter, format_data_short
    """
    output_var = input_var.format_data_short()

def matplotlib_ticker_FuncFormatter_format_ticks(self: 'any') -> 'any_computed':
    r"""
    Return the tick labels for all the ticks at once.
    Keywords: matplotlib.ticker.FuncFormatter, format_ticks
    """
    output_var = input_var.format_ticks()

def matplotlib_ticker_FuncFormatter_get_offset(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.ticker.FuncFormatter, get_offset
    """
    output_var = input_var.get_offset()

def matplotlib_ticker_FuncFormatter_set_axis(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.ticker.FuncFormatter, set_axis
    """
    output_var = input_var.set_axis()

def matplotlib_ticker_FuncFormatter_set_locs(self: 'any') -> 'any_computed':
    r"""
    Set the locations of the ticks.
    This method is called before computing the tick labels because some
    formatters need to know all tick locations to do so.
    Keywords: matplotlib.ticker.FuncFormatter, set_locs
    """
    output_var = input_var.set_locs()

def matplotlib_ticker_FuncFormatter_set_offset_string(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.ticker.FuncFormatter, set_offset_string
    """
    output_var = input_var.set_offset_string()

def matplotlib_ticker_IndexLocator_create_dummy_axis(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.ticker.IndexLocator, create_dummy_axis
    """
    output_var = input_var.create_dummy_axis()

def matplotlib_ticker_IndexLocator_nonsingular(self: 'any') -> 'any_computed':
    r"""
    Adjust a range as needed to avoid singularities.
    This method gets called during autoscaling, with ``(v0, v1)`` set to
    the data limits on the Axes if the Axes contains any data, or
    ``(-inf, +inf)`` if not.
    Keywords: matplotlib.ticker.IndexLocator, nonsingular
    """
    output_var = input_var.nonsingular()

def matplotlib_ticker_IndexLocator_raise_if_exceeds(self: 'any') -> 'any_computed':
    r"""
    Log at WARNING level if *locs* is longer than `Locator.MAXTICKS`.
    This is intended to be called immediately before returning *locs* from
    ``__call__`` to inform users in case their Locator returns a huge
    number of ticks, causing Matplotlib to run out of memory.
    Keywords: matplotlib.ticker.IndexLocator, raise_if_exceeds
    """
    output_var = input_var.raise_if_exceeds()

def matplotlib_ticker_IndexLocator_set_axis(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.ticker.IndexLocator, set_axis
    """
    output_var = input_var.set_axis()

def matplotlib_ticker_IndexLocator_set_params(self: 'any') -> 'any_computed':
    r"""
    Set parameters within this locator
    Keywords: matplotlib.ticker.IndexLocator, set_params
    """
    output_var = input_var.set_params()

def matplotlib_ticker_IndexLocator_tick_values(self: 'any') -> 'any_computed':
    r"""
    Return the values of the located ticks given **vmin** and **vmax**.
    .. note::
        To get tick locations with the vmin and vmax values defined
        automatically for the associated ``axis`` simply call
    Keywords: matplotlib.ticker.IndexLocator, tick_values
    """
    output_var = input_var.tick_values()

def matplotlib_ticker_IndexLocator_view_limits(self: 'any') -> 'any_computed':
    r"""
    Select a scale for the range from vmin to vmax.
    Subclasses should override this method to change locator behaviour.
    Keywords: matplotlib.ticker.IndexLocator, view_limits
    """
    output_var = input_var.view_limits()

def matplotlib_ticker_LinearLocator_create_dummy_axis(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.ticker.LinearLocator, create_dummy_axis
    """
    output_var = input_var.create_dummy_axis()

def matplotlib_ticker_LinearLocator_nonsingular(self: 'any') -> 'any_computed':
    r"""
    Adjust a range as needed to avoid singularities.
    This method gets called during autoscaling, with ``(v0, v1)`` set to
    the data limits on the Axes if the Axes contains any data, or
    ``(-inf, +inf)`` if not.
    Keywords: matplotlib.ticker.LinearLocator, nonsingular
    """
    output_var = input_var.nonsingular()

def matplotlib_ticker_LinearLocator_raise_if_exceeds(self: 'any') -> 'any_computed':
    r"""
    Log at WARNING level if *locs* is longer than `Locator.MAXTICKS`.
    This is intended to be called immediately before returning *locs* from
    ``__call__`` to inform users in case their Locator returns a huge
    number of ticks, causing Matplotlib to run out of memory.
    Keywords: matplotlib.ticker.LinearLocator, raise_if_exceeds
    """
    output_var = input_var.raise_if_exceeds()

def matplotlib_ticker_LinearLocator_set_axis(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.ticker.LinearLocator, set_axis
    """
    output_var = input_var.set_axis()

def matplotlib_ticker_LinearLocator_set_params(self: 'any') -> 'any_computed':
    r"""
    Set parameters within this locator.
    Keywords: matplotlib.ticker.LinearLocator, set_params
    """
    output_var = input_var.set_params()

def matplotlib_ticker_LinearLocator_tick_values(self: 'any') -> 'any_computed':
    r"""
    Return the values of the located ticks given **vmin** and **vmax**.
    .. note::
        To get tick locations with the vmin and vmax values defined
        automatically for the associated ``axis`` simply call
    Keywords: matplotlib.ticker.LinearLocator, tick_values
    """
    output_var = input_var.tick_values()

def matplotlib_ticker_LinearLocator_view_limits(self: 'any') -> 'any_computed':
    r"""
    Try to choose the view limits intelligently.
    Keywords: matplotlib.ticker.LinearLocator, view_limits
    """
    output_var = input_var.view_limits()

def matplotlib_ticker_Locator_create_dummy_axis(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.ticker.Locator, create_dummy_axis
    """
    output_var = input_var.create_dummy_axis()

def matplotlib_ticker_Locator_nonsingular(self: 'any') -> 'any_computed':
    r"""
    Adjust a range as needed to avoid singularities.
    This method gets called during autoscaling, with ``(v0, v1)`` set to
    the data limits on the Axes if the Axes contains any data, or
    ``(-inf, +inf)`` if not.
    Keywords: matplotlib.ticker.Locator, nonsingular
    """
    output_var = input_var.nonsingular()

def matplotlib_ticker_Locator_raise_if_exceeds(self: 'any') -> 'any_computed':
    r"""
    Log at WARNING level if *locs* is longer than `Locator.MAXTICKS`.
    This is intended to be called immediately before returning *locs* from
    ``__call__`` to inform users in case their Locator returns a huge
    number of ticks, causing Matplotlib to run out of memory.
    Keywords: matplotlib.ticker.Locator, raise_if_exceeds
    """
    output_var = input_var.raise_if_exceeds()

def matplotlib_ticker_Locator_set_axis(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.ticker.Locator, set_axis
    """
    output_var = input_var.set_axis()

def matplotlib_ticker_Locator_set_params(self: 'any') -> 'any_computed':
    r"""
    Do nothing, and raise a warning. Any locator class not supporting the
    set_params() function will call this.
    Keywords: matplotlib.ticker.Locator, set_params
    """
    output_var = input_var.set_params()

def matplotlib_ticker_Locator_tick_values(self: 'any') -> 'any_computed':
    r"""
    Return the values of the located ticks given **vmin** and **vmax**.
    .. note::
        To get tick locations with the vmin and vmax values defined
        automatically for the associated ``axis`` simply call
    Keywords: matplotlib.ticker.Locator, tick_values
    """
    output_var = input_var.tick_values()

def matplotlib_ticker_Locator_view_limits(self: 'any') -> 'any_computed':
    r"""
    Select a scale for the range from vmin to vmax.
    Subclasses should override this method to change locator behaviour.
    Keywords: matplotlib.ticker.Locator, view_limits
    """
    output_var = input_var.view_limits()

def matplotlib_ticker_LogFormatter_create_dummy_axis(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.ticker.LogFormatter, create_dummy_axis
    """
    output_var = input_var.create_dummy_axis()

def matplotlib_ticker_LogFormatter_fix_minus(s: 'any') -> 'any_computed':
    r"""
    Some classes may want to replace a hyphen for minus with the proper
    Unicode symbol (U+2212) for typographical correctness.  This is a
    helper method to perform such a replacement when it is enabled via
    :rc:`axes.unicode_minus`.
    Keywords: matplotlib.ticker.LogFormatter, fix_minus
    """
    output_var = input_var.fix_minus()

def matplotlib_ticker_LogFormatter_format_data(self: 'any') -> 'any_computed':
    r"""
    Return the full string representation of the value with the
    position unspecified.
    Keywords: matplotlib.ticker.LogFormatter, format_data
    """
    output_var = input_var.format_data()

def matplotlib_ticker_LogFormatter_format_data_short(self: 'any') -> 'any_computed':
    r"""
    Return a short string version of the tick value.
    Defaults to the position-independent long value.
    Keywords: matplotlib.ticker.LogFormatter, format_data_short
    """
    output_var = input_var.format_data_short()

def matplotlib_ticker_LogFormatter_format_ticks(self: 'any') -> 'any_computed':
    r"""
    Return the tick labels for all the ticks at once.
    Keywords: matplotlib.ticker.LogFormatter, format_ticks
    """
    output_var = input_var.format_ticks()

def matplotlib_ticker_LogFormatter_get_offset(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.ticker.LogFormatter, get_offset
    """
    output_var = input_var.get_offset()

def matplotlib_ticker_LogFormatter_set_axis(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.ticker.LogFormatter, set_axis
    """
    output_var = input_var.set_axis()

def matplotlib_ticker_LogFormatter_set_base(self: 'any') -> 'any_computed':
    r"""
    Change the *base* for labeling.
    .. warning::
       Should always match the base used for :class:`LogLocator`
    Keywords: matplotlib.ticker.LogFormatter, set_base
    """
    output_var = input_var.set_base()

def matplotlib_ticker_LogFormatter_set_label_minor(self: 'any') -> 'any_computed':
    r"""
    Switch minor tick labeling on or off.
    Parameters
    ----------
    labelOnlyBase : bool
    Keywords: matplotlib.ticker.LogFormatter, set_label_minor
    """
    output_var = input_var.set_label_minor()

def matplotlib_ticker_LogFormatter_set_locs(self: 'any') -> 'any_computed':
    r"""
    Use axis view limits to control which ticks are labeled.
    The *locs* parameter is ignored in the present algorithm.
    Keywords: matplotlib.ticker.LogFormatter, set_locs
    """
    output_var = input_var.set_locs()

def matplotlib_ticker_LogFormatterExponent_create_dummy_axis(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.ticker.LogFormatterExponent, create_dummy_axis
    """
    output_var = input_var.create_dummy_axis()

def matplotlib_ticker_LogFormatterExponent_fix_minus(s: 'any') -> 'any_computed':
    r"""
    Some classes may want to replace a hyphen for minus with the proper
    Unicode symbol (U+2212) for typographical correctness.  This is a
    helper method to perform such a replacement when it is enabled via
    :rc:`axes.unicode_minus`.
    Keywords: matplotlib.ticker.LogFormatterExponent, fix_minus
    """
    output_var = input_var.fix_minus()

def matplotlib_ticker_LogFormatterExponent_format_data(self: 'any') -> 'any_computed':
    r"""
    Return the full string representation of the value with the
    position unspecified.
    Keywords: matplotlib.ticker.LogFormatterExponent, format_data
    """
    output_var = input_var.format_data()

def matplotlib_ticker_LogFormatterExponent_format_data_short(self: 'any') -> 'any_computed':
    r"""
    Return a short string version of the tick value.
    Defaults to the position-independent long value.
    Keywords: matplotlib.ticker.LogFormatterExponent, format_data_short
    """
    output_var = input_var.format_data_short()

def matplotlib_ticker_LogFormatterExponent_format_ticks(self: 'any') -> 'any_computed':
    r"""
    Return the tick labels for all the ticks at once.
    Keywords: matplotlib.ticker.LogFormatterExponent, format_ticks
    """
    output_var = input_var.format_ticks()

def matplotlib_ticker_LogFormatterExponent_get_offset(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.ticker.LogFormatterExponent, get_offset
    """
    output_var = input_var.get_offset()

def matplotlib_ticker_LogFormatterExponent_set_axis(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.ticker.LogFormatterExponent, set_axis
    """
    output_var = input_var.set_axis()

def matplotlib_ticker_LogFormatterExponent_set_base(self: 'any') -> 'any_computed':
    r"""
    Change the *base* for labeling.
    .. warning::
       Should always match the base used for :class:`LogLocator`
    Keywords: matplotlib.ticker.LogFormatterExponent, set_base
    """
    output_var = input_var.set_base()

def matplotlib_ticker_LogFormatterExponent_set_label_minor(self: 'any') -> 'any_computed':
    r"""
    Switch minor tick labeling on or off.
    Parameters
    ----------
    labelOnlyBase : bool
    Keywords: matplotlib.ticker.LogFormatterExponent, set_label_minor
    """
    output_var = input_var.set_label_minor()

def matplotlib_ticker_LogFormatterExponent_set_locs(self: 'any') -> 'any_computed':
    r"""
    Use axis view limits to control which ticks are labeled.
    The *locs* parameter is ignored in the present algorithm.
    Keywords: matplotlib.ticker.LogFormatterExponent, set_locs
    """
    output_var = input_var.set_locs()

def matplotlib_ticker_LogFormatterMathtext_create_dummy_axis(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.ticker.LogFormatterMathtext, create_dummy_axis
    """
    output_var = input_var.create_dummy_axis()

def matplotlib_ticker_LogFormatterMathtext_fix_minus(s: 'any') -> 'any_computed':
    r"""
    Some classes may want to replace a hyphen for minus with the proper
    Unicode symbol (U+2212) for typographical correctness.  This is a
    helper method to perform such a replacement when it is enabled via
    :rc:`axes.unicode_minus`.
    Keywords: matplotlib.ticker.LogFormatterMathtext, fix_minus
    """
    output_var = input_var.fix_minus()

def matplotlib_ticker_LogFormatterMathtext_format_data(self: 'any') -> 'any_computed':
    r"""
    Return the full string representation of the value with the
    position unspecified.
    Keywords: matplotlib.ticker.LogFormatterMathtext, format_data
    """
    output_var = input_var.format_data()

def matplotlib_ticker_LogFormatterMathtext_format_data_short(self: 'any') -> 'any_computed':
    r"""
    Return a short string version of the tick value.
    Defaults to the position-independent long value.
    Keywords: matplotlib.ticker.LogFormatterMathtext, format_data_short
    """
    output_var = input_var.format_data_short()

def matplotlib_ticker_LogFormatterMathtext_format_ticks(self: 'any') -> 'any_computed':
    r"""
    Return the tick labels for all the ticks at once.
    Keywords: matplotlib.ticker.LogFormatterMathtext, format_ticks
    """
    output_var = input_var.format_ticks()

def matplotlib_ticker_LogFormatterMathtext_get_offset(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.ticker.LogFormatterMathtext, get_offset
    """
    output_var = input_var.get_offset()

def matplotlib_ticker_LogFormatterMathtext_set_axis(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.ticker.LogFormatterMathtext, set_axis
    """
    output_var = input_var.set_axis()

def matplotlib_ticker_LogFormatterMathtext_set_base(self: 'any') -> 'any_computed':
    r"""
    Change the *base* for labeling.
    .. warning::
       Should always match the base used for :class:`LogLocator`
    Keywords: matplotlib.ticker.LogFormatterMathtext, set_base
    """
    output_var = input_var.set_base()

def matplotlib_ticker_LogFormatterMathtext_set_label_minor(self: 'any') -> 'any_computed':
    r"""
    Switch minor tick labeling on or off.
    Parameters
    ----------
    labelOnlyBase : bool
    Keywords: matplotlib.ticker.LogFormatterMathtext, set_label_minor
    """
    output_var = input_var.set_label_minor()

def matplotlib_ticker_LogFormatterMathtext_set_locs(self: 'any') -> 'any_computed':
    r"""
    Use axis view limits to control which ticks are labeled.
    The *locs* parameter is ignored in the present algorithm.
    Keywords: matplotlib.ticker.LogFormatterMathtext, set_locs
    """
    output_var = input_var.set_locs()

def matplotlib_ticker_MaxNLocator_create_dummy_axis(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.ticker.MaxNLocator, create_dummy_axis
    """
    output_var = input_var.create_dummy_axis()

def matplotlib_ticker_MaxNLocator_nonsingular(self: 'any') -> 'any_computed':
    r"""
    Adjust a range as needed to avoid singularities.
    This method gets called during autoscaling, with ``(v0, v1)`` set to
    the data limits on the Axes if the Axes contains any data, or
    ``(-inf, +inf)`` if not.
    Keywords: matplotlib.ticker.MaxNLocator, nonsingular
    """
    output_var = input_var.nonsingular()

def matplotlib_ticker_MaxNLocator_raise_if_exceeds(self: 'any') -> 'any_computed':
    r"""
    Log at WARNING level if *locs* is longer than `Locator.MAXTICKS`.
    This is intended to be called immediately before returning *locs* from
    ``__call__`` to inform users in case their Locator returns a huge
    number of ticks, causing Matplotlib to run out of memory.
    Keywords: matplotlib.ticker.MaxNLocator, raise_if_exceeds
    """
    output_var = input_var.raise_if_exceeds()

def matplotlib_ticker_MaxNLocator_set_axis(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.ticker.MaxNLocator, set_axis
    """
    output_var = input_var.set_axis()

def matplotlib_ticker_MaxNLocator_set_params(self: 'any') -> 'any_computed':
    r"""
    Set parameters for this locator.
    Parameters
    ----------
    nbins : int or 'auto', optional
    Keywords: matplotlib.ticker.MaxNLocator, set_params
    """
    output_var = input_var.set_params()

def matplotlib_ticker_MaxNLocator_tick_values(self: 'any') -> 'any_computed':
    r"""
    Return the values of the located ticks given **vmin** and **vmax**.
    .. note::
        To get tick locations with the vmin and vmax values defined
        automatically for the associated ``axis`` simply call
    Keywords: matplotlib.ticker.MaxNLocator, tick_values
    """
    output_var = input_var.tick_values()

def matplotlib_ticker_MaxNLocator_view_limits(self: 'any') -> 'any_computed':
    r"""
    Select a scale for the range from vmin to vmax.
    Subclasses should override this method to change locator behaviour.
    Keywords: matplotlib.ticker.MaxNLocator, view_limits
    """
    output_var = input_var.view_limits()

def matplotlib_ticker_MultipleLocator_create_dummy_axis(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.ticker.MultipleLocator, create_dummy_axis
    """
    output_var = input_var.create_dummy_axis()

def matplotlib_ticker_MultipleLocator_nonsingular(self: 'any') -> 'any_computed':
    r"""
    Adjust a range as needed to avoid singularities.
    This method gets called during autoscaling, with ``(v0, v1)`` set to
    the data limits on the Axes if the Axes contains any data, or
    ``(-inf, +inf)`` if not.
    Keywords: matplotlib.ticker.MultipleLocator, nonsingular
    """
    output_var = input_var.nonsingular()

def matplotlib_ticker_MultipleLocator_raise_if_exceeds(self: 'any') -> 'any_computed':
    r"""
    Log at WARNING level if *locs* is longer than `Locator.MAXTICKS`.
    This is intended to be called immediately before returning *locs* from
    ``__call__`` to inform users in case their Locator returns a huge
    number of ticks, causing Matplotlib to run out of memory.
    Keywords: matplotlib.ticker.MultipleLocator, raise_if_exceeds
    """
    output_var = input_var.raise_if_exceeds()

def matplotlib_ticker_MultipleLocator_set_axis(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.ticker.MultipleLocator, set_axis
    """
    output_var = input_var.set_axis()

def matplotlib_ticker_MultipleLocator_set_params(self: 'any') -> 'any_computed':
    r"""
    Set parameters within this locator.
    Parameters
    ----------
    base : float > 0, optional
    Keywords: matplotlib.ticker.MultipleLocator, set_params
    """
    output_var = input_var.set_params()

def matplotlib_ticker_MultipleLocator_tick_values(self: 'any') -> 'any_computed':
    r"""
    Return the values of the located ticks given **vmin** and **vmax**.
    .. note::
        To get tick locations with the vmin and vmax values defined
        automatically for the associated ``axis`` simply call
    Keywords: matplotlib.ticker.MultipleLocator, tick_values
    """
    output_var = input_var.tick_values()

def matplotlib_ticker_MultipleLocator_view_limits(self: 'any') -> 'any_computed':
    r"""
    Set the view limits to the nearest tick values that contain the data.
    Keywords: matplotlib.ticker.MultipleLocator, view_limits
    """
    output_var = input_var.view_limits()

def matplotlib_ticker_PercentFormatter_convert_to_pct(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.ticker.PercentFormatter, convert_to_pct
    """
    output_var = input_var.convert_to_pct()

def matplotlib_ticker_PercentFormatter_create_dummy_axis(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.ticker.PercentFormatter, create_dummy_axis
    """
    output_var = input_var.create_dummy_axis()

def matplotlib_ticker_PercentFormatter_fix_minus(s: 'any') -> 'any_computed':
    r"""
    Some classes may want to replace a hyphen for minus with the proper
    Unicode symbol (U+2212) for typographical correctness.  This is a
    helper method to perform such a replacement when it is enabled via
    :rc:`axes.unicode_minus`.
    Keywords: matplotlib.ticker.PercentFormatter, fix_minus
    """
    output_var = input_var.fix_minus()

def matplotlib_ticker_PercentFormatter_format_data(self: 'any') -> 'any_computed':
    r"""
    Return the full string representation of the value with the
    position unspecified.
    Keywords: matplotlib.ticker.PercentFormatter, format_data
    """
    output_var = input_var.format_data()

def matplotlib_ticker_PercentFormatter_format_data_short(self: 'any') -> 'any_computed':
    r"""
    Return a short string version of the tick value.
    Defaults to the position-independent long value.
    Keywords: matplotlib.ticker.PercentFormatter, format_data_short
    """
    output_var = input_var.format_data_short()

def matplotlib_ticker_PercentFormatter_format_pct(self: 'any') -> 'any_computed':
    r"""
    Format the number as a percentage number with the correct
    number of decimals and adds the percent symbol, if any.
    If ``self.decimals`` is `None`, the number of digits after the
    decimal point is set based on the *display_range* of the axis
    Keywords: matplotlib.ticker.PercentFormatter, format_pct
    """
    output_var = input_var.format_pct()

def matplotlib_ticker_PercentFormatter_format_ticks(self: 'any') -> 'any_computed':
    r"""
    Return the tick labels for all the ticks at once.
    Keywords: matplotlib.ticker.PercentFormatter, format_ticks
    """
    output_var = input_var.format_ticks()

def matplotlib_ticker_PercentFormatter_get_offset(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.ticker.PercentFormatter, get_offset
    """
    output_var = input_var.get_offset()

def matplotlib_ticker_PercentFormatter_set_axis(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.ticker.PercentFormatter, set_axis
    """
    output_var = input_var.set_axis()

def matplotlib_ticker_PercentFormatter_set_locs(self: 'any') -> 'any_computed':
    r"""
    Set the locations of the ticks.
    This method is called before computing the tick labels because some
    formatters need to know all tick locations to do so.
    Keywords: matplotlib.ticker.PercentFormatter, set_locs
    """
    output_var = input_var.set_locs()

def matplotlib_ticker_StrMethodFormatter_create_dummy_axis(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.ticker.StrMethodFormatter, create_dummy_axis
    """
    output_var = input_var.create_dummy_axis()

def matplotlib_ticker_StrMethodFormatter_fix_minus(s: 'any') -> 'any_computed':
    r"""
    Some classes may want to replace a hyphen for minus with the proper
    Unicode symbol (U+2212) for typographical correctness.  This is a
    helper method to perform such a replacement when it is enabled via
    :rc:`axes.unicode_minus`.
    Keywords: matplotlib.ticker.StrMethodFormatter, fix_minus
    """
    output_var = input_var.fix_minus()

def matplotlib_ticker_StrMethodFormatter_format_data(self: 'any') -> 'any_computed':
    r"""
    Return the full string representation of the value with the
    position unspecified.
    Keywords: matplotlib.ticker.StrMethodFormatter, format_data
    """
    output_var = input_var.format_data()

def matplotlib_ticker_StrMethodFormatter_format_data_short(self: 'any') -> 'any_computed':
    r"""
    Return a short string version of the tick value.
    Defaults to the position-independent long value.
    Keywords: matplotlib.ticker.StrMethodFormatter, format_data_short
    """
    output_var = input_var.format_data_short()

def matplotlib_ticker_StrMethodFormatter_format_ticks(self: 'any') -> 'any_computed':
    r"""
    Return the tick labels for all the ticks at once.
    Keywords: matplotlib.ticker.StrMethodFormatter, format_ticks
    """
    output_var = input_var.format_ticks()

def matplotlib_ticker_StrMethodFormatter_get_offset(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.ticker.StrMethodFormatter, get_offset
    """
    output_var = input_var.get_offset()

def matplotlib_ticker_StrMethodFormatter_set_axis(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.ticker.StrMethodFormatter, set_axis
    """
    output_var = input_var.set_axis()

def matplotlib_ticker_StrMethodFormatter_set_locs(self: 'any') -> 'any_computed':
    r"""
    Set the locations of the ticks.
    This method is called before computing the tick labels because some
    formatters need to know all tick locations to do so.
    Keywords: matplotlib.ticker.StrMethodFormatter, set_locs
    """
    output_var = input_var.set_locs()

def matplotlib_ticker_TickHelper_create_dummy_axis(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.ticker.TickHelper, create_dummy_axis
    """
    output_var = input_var.create_dummy_axis()

def matplotlib_ticker_TickHelper_set_axis(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.ticker.TickHelper, set_axis
    """
    output_var = input_var.set_axis()

def matplotlib_transforms_Affine2D_clear(self: 'any') -> 'any_computed':
    r"""
    Reset the underlying matrix to the identity transform.
    Keywords: matplotlib.transforms.Affine2D, clear
    """
    output_var = input_var.clear()

def matplotlib_transforms_Affine2D_contains_branch(self: 'any') -> 'any_computed':
    r"""
    Return whether the given transform is a sub-tree of this transform.
    This routine uses transform equality to identify sub-trees, therefore
    in many situations it is object id which will be used.
    Keywords: matplotlib.transforms.Affine2D, contains_branch
    """
    output_var = input_var.contains_branch()

def matplotlib_transforms_Affine2D_contains_branch_seperately(self: 'any') -> 'any_computed':
    r"""
    Return whether the given branch is a sub-tree of this transform on
    each separate dimension.
    A common use for this method is to identify if a transform is a blended
    transform containing an Axes' data transform. e.g.::
    Keywords: matplotlib.transforms.Affine2D, contains_branch_seperately
    """
    output_var = input_var.contains_branch_seperately()

def matplotlib_transforms_Affine2D_from_values(a: 'any') -> 'any_computed':
    r"""
    Create a new Affine2D instance from the given values::
      a c e
      b d f
      0 0 1
    Keywords: matplotlib.transforms.Affine2D, from_values
    """
    output_var = input_var.from_values()

def matplotlib_transforms_Affine2D_frozen(self: 'any') -> 'any_computed':
    r"""
    Return a frozen copy of this transform node.  The frozen copy will not
    be updated when its children change.  Useful for storing a previously
    known state of a transform where ``copy.deepcopy()`` might normally be
    used.
    Keywords: matplotlib.transforms.Affine2D, frozen
    """
    output_var = input_var.frozen()

def matplotlib_transforms_Affine2D_get_affine(self: 'any') -> 'any_computed':
    r"""
    Get the affine part of this transform.
    Keywords: matplotlib.transforms.Affine2D, get_affine
    """
    output_var = input_var.get_affine()

def matplotlib_transforms_Affine2D_get_matrix(self: 'any') -> 'any_computed':
    r"""
    Get the underlying transformation matrix as a 3x3 array::
      a c e
      b d f
      0 0 1
    Keywords: matplotlib.transforms.Affine2D, get_matrix
    """
    output_var = input_var.get_matrix()

def matplotlib_transforms_Affine2D_invalidate(self: 'any') -> 'any_computed':
    r"""
    Invalidate this `TransformNode` and triggers an invalidation of its
    ancestors.  Should be called any time the transform changes.
    Keywords: matplotlib.transforms.Affine2D, invalidate
    """
    output_var = input_var.invalidate()

def matplotlib_transforms_Affine2D_inverted(self: 'any') -> 'any_computed':
    r"""
    Return the corresponding inverse transformation.
    It holds ``x == self.inverted().transform(self.transform(x))``.
    The return value of this method should be treated as
    Keywords: matplotlib.transforms.Affine2D, inverted
    """
    output_var = input_var.inverted()

def matplotlib_transforms_Affine2D_rotate(self: 'any') -> 'any_computed':
    r"""
    Add a rotation (in radians) to this transform in place.
    Returns *self*, so this method can easily be chained with more
    calls to :meth:`rotate`, :meth:`rotate_deg`, :meth:`translate`
    and :meth:`scale`.
    Keywords: matplotlib.transforms.Affine2D, rotate
    """
    output_var = input_var.rotate()

def matplotlib_transforms_Affine2D_rotate_around(self: 'any') -> 'any_computed':
    r"""
    Add a rotation (in radians) around the point (x, y) in place.
    Returns *self*, so this method can easily be chained with more
    calls to :meth:`rotate`, :meth:`rotate_deg`, :meth:`translate`
    and :meth:`scale`.
    Keywords: matplotlib.transforms.Affine2D, rotate_around
    """
    output_var = input_var.rotate_around()

def matplotlib_transforms_Affine2D_rotate_deg(self: 'any') -> 'any_computed':
    r"""
    Add a rotation (in degrees) to this transform in place.
    Returns *self*, so this method can easily be chained with more
    calls to :meth:`rotate`, :meth:`rotate_deg`, :meth:`translate`
    and :meth:`scale`.
    Keywords: matplotlib.transforms.Affine2D, rotate_deg
    """
    output_var = input_var.rotate_deg()

def matplotlib_transforms_Affine2D_rotate_deg_around(self: 'any') -> 'any_computed':
    r"""
    Add a rotation (in degrees) around the point (x, y) in place.
    Returns *self*, so this method can easily be chained with more
    calls to :meth:`rotate`, :meth:`rotate_deg`, :meth:`translate`
    and :meth:`scale`.
    Keywords: matplotlib.transforms.Affine2D, rotate_deg_around
    """
    output_var = input_var.rotate_deg_around()

def matplotlib_transforms_Affine2D_scale(self: 'any') -> 'any_computed':
    r"""
    Add a scale in place.
    If *sy* is None, the same scale is applied in both the *x*- and
    *y*-directions.
    Keywords: matplotlib.transforms.Affine2D, scale
    """
    output_var = input_var.scale()

def matplotlib_transforms_Affine2D_set(self: 'any') -> 'any_computed':
    r"""
    Set this transformation from the frozen copy of another
    `Affine2DBase` object.
    Keywords: matplotlib.transforms.Affine2D, set
    """
    output_var = input_var.set()

def matplotlib_transforms_Affine2D_set_children(self: 'any') -> 'any_computed':
    r"""
    Set the children of the transform, to let the invalidation
    system know which transforms can invalidate this transform.
    Should be called from the constructor of any transforms that
    depend on other transforms.
    Keywords: matplotlib.transforms.Affine2D, set_children
    """
    output_var = input_var.set_children()

def matplotlib_transforms_Affine2D_set_matrix(self: 'any') -> 'any_computed':
    r"""
    Set the underlying transformation matrix from a 3x3 array::
      a c e
      b d f
      0 0 1
    Keywords: matplotlib.transforms.Affine2D, set_matrix
    """
    output_var = input_var.set_matrix()

def matplotlib_transforms_Affine2D_skew(self: 'any') -> 'any_computed':
    r"""
    Add a skew in place.
    *xShear* and *yShear* are the shear angles along the *x*- and
    *y*-axes, respectively, in radians.
    Keywords: matplotlib.transforms.Affine2D, skew
    """
    output_var = input_var.skew()

def matplotlib_transforms_Affine2D_skew_deg(self: 'any') -> 'any_computed':
    r"""
    Add a skew in place.
    *xShear* and *yShear* are the shear angles along the *x*- and
    *y*-axes, respectively, in degrees.
    Keywords: matplotlib.transforms.Affine2D, skew_deg
    """
    output_var = input_var.skew_deg()

def matplotlib_transforms_Affine2D_to_values(self: 'any') -> 'any_computed':
    r"""
    Return the values of the matrix as an ``(a, b, c, d, e, f)`` tuple.
    Keywords: matplotlib.transforms.Affine2D, to_values
    """
    output_var = input_var.to_values()

def matplotlib_transforms_Affine2D_transform(self: 'any') -> 'any_computed':
    r"""
    Apply this transformation on the given array of *values*.
    Parameters
    ----------
    values : array-like
    Keywords: matplotlib.transforms.Affine2D, transform
    """
    output_var = input_var.transform()

def matplotlib_transforms_Affine2D_transform_affine(self: 'any') -> 'any_computed':
    r"""
    Apply only the affine part of this transformation on the
    given array of values.
    ``transform(values)`` is always equivalent to
    ``transform_affine(transform_non_affine(values))``.
    Keywords: matplotlib.transforms.Affine2D, transform_affine
    """
    output_var = input_var.transform_affine()

def matplotlib_transforms_Affine2D_transform_angles(self: 'any') -> 'any_computed':
    r"""
    Transform a set of angles anchored at specific locations.
    Parameters
    ----------
    angles : (N,) array-like
    Keywords: matplotlib.transforms.Affine2D, transform_angles
    """
    output_var = input_var.transform_angles()

def matplotlib_transforms_Affine2D_transform_bbox(self: 'any') -> 'any_computed':
    r"""
    Transform the given bounding box.
    For smarter transforms including caching (a common requirement in
    Matplotlib), see `TransformedBbox`.
    Keywords: matplotlib.transforms.Affine2D, transform_bbox
    """
    output_var = input_var.transform_bbox()

def matplotlib_transforms_Affine2D_transform_non_affine(self: 'any') -> 'any_computed':
    r"""
    Apply only the non-affine part of this transformation.
    ``transform(values)`` is always equivalent to
    ``transform_affine(transform_non_affine(values))``.
    Keywords: matplotlib.transforms.Affine2D, transform_non_affine
    """
    output_var = input_var.transform_non_affine()

def matplotlib_transforms_Affine2D_transform_path(self: 'any') -> 'any_computed':
    r"""
    Apply the transform to `.Path` *path*, returning a new `.Path`.
    In some cases, this transform may insert curves into the path
    that began as line segments.
    Keywords: matplotlib.transforms.Affine2D, transform_path
    """
    output_var = input_var.transform_path()

def matplotlib_transforms_Affine2D_transform_path_affine(self: 'any') -> 'any_computed':
    r"""
    Apply the affine part of this transform to `.Path` *path*, returning a
    new `.Path`.
    ``transform_path(path)`` is equivalent to
    ``transform_path_affine(transform_path_non_affine(values))``.
    Keywords: matplotlib.transforms.Affine2D, transform_path_affine
    """
    output_var = input_var.transform_path_affine()

def matplotlib_transforms_Affine2D_transform_path_non_affine(self: 'any') -> 'any_computed':
    r"""
    Apply the non-affine part of this transform to `.Path` *path*,
    returning a new `.Path`.
    ``transform_path(path)`` is equivalent to
    ``transform_path_affine(transform_path_non_affine(values))``.
    Keywords: matplotlib.transforms.Affine2D, transform_path_non_affine
    """
    output_var = input_var.transform_path_non_affine()

def matplotlib_transforms_Affine2D_transform_point(self: 'any') -> 'any_computed':
    r"""
    Return a transformed point.
    This function is only kept for backcompatibility; the more general
    `.transform` method is capable of transforming both a list of points
    and a single point.
    Keywords: matplotlib.transforms.Affine2D, transform_point
    """
    output_var = input_var.transform_point()

def matplotlib_transforms_Affine2D_translate(self: 'any') -> 'any_computed':
    r"""
    Add a translation in place.
    Returns *self*, so this method can easily be chained with more
    calls to :meth:`rotate`, :meth:`rotate_deg`, :meth:`translate`
    and :meth:`scale`.
    Keywords: matplotlib.transforms.Affine2D, translate
    """
    output_var = input_var.translate()

def matplotlib_transforms_Affine2DBase_contains_branch(self: 'any') -> 'any_computed':
    r"""
    Return whether the given transform is a sub-tree of this transform.
    This routine uses transform equality to identify sub-trees, therefore
    in many situations it is object id which will be used.
    Keywords: matplotlib.transforms.Affine2DBase, contains_branch
    """
    output_var = input_var.contains_branch()

def matplotlib_transforms_Affine2DBase_contains_branch_seperately(self: 'any') -> 'any_computed':
    r"""
    Return whether the given branch is a sub-tree of this transform on
    each separate dimension.
    A common use for this method is to identify if a transform is a blended
    transform containing an Axes' data transform. e.g.::
    Keywords: matplotlib.transforms.Affine2DBase, contains_branch_seperately
    """
    output_var = input_var.contains_branch_seperately()

def matplotlib_transforms_Affine2DBase_frozen(self: 'any') -> 'any_computed':
    r"""
    Return a frozen copy of this transform node.  The frozen copy will not
    be updated when its children change.  Useful for storing a previously
    known state of a transform where ``copy.deepcopy()`` might normally be
    used.
    Keywords: matplotlib.transforms.Affine2DBase, frozen
    """
    output_var = input_var.frozen()

def matplotlib_transforms_Affine2DBase_get_affine(self: 'any') -> 'any_computed':
    r"""
    Get the affine part of this transform.
    Keywords: matplotlib.transforms.Affine2DBase, get_affine
    """
    output_var = input_var.get_affine()

def matplotlib_transforms_Affine2DBase_get_matrix(self: 'any') -> 'any_computed':
    r"""
    Get the matrix for the affine part of this transform.
    Keywords: matplotlib.transforms.Affine2DBase, get_matrix
    """
    output_var = input_var.get_matrix()

def matplotlib_transforms_Affine2DBase_invalidate(self: 'any') -> 'any_computed':
    r"""
    Invalidate this `TransformNode` and triggers an invalidation of its
    ancestors.  Should be called any time the transform changes.
    Keywords: matplotlib.transforms.Affine2DBase, invalidate
    """
    output_var = input_var.invalidate()

def matplotlib_transforms_Affine2DBase_inverted(self: 'any') -> 'any_computed':
    r"""
    Return the corresponding inverse transformation.
    It holds ``x == self.inverted().transform(self.transform(x))``.
    The return value of this method should be treated as
    Keywords: matplotlib.transforms.Affine2DBase, inverted
    """
    output_var = input_var.inverted()

def matplotlib_transforms_Affine2DBase_set_children(self: 'any') -> 'any_computed':
    r"""
    Set the children of the transform, to let the invalidation
    system know which transforms can invalidate this transform.
    Should be called from the constructor of any transforms that
    depend on other transforms.
    Keywords: matplotlib.transforms.Affine2DBase, set_children
    """
    output_var = input_var.set_children()

def matplotlib_transforms_Affine2DBase_to_values(self: 'any') -> 'any_computed':
    r"""
    Return the values of the matrix as an ``(a, b, c, d, e, f)`` tuple.
    Keywords: matplotlib.transforms.Affine2DBase, to_values
    """
    output_var = input_var.to_values()

def matplotlib_transforms_Affine2DBase_transform(self: 'any') -> 'any_computed':
    r"""
    Apply this transformation on the given array of *values*.
    Parameters
    ----------
    values : array-like
    Keywords: matplotlib.transforms.Affine2DBase, transform
    """
    output_var = input_var.transform()

def matplotlib_transforms_Affine2DBase_transform_affine(self: 'any') -> 'any_computed':
    r"""
    Apply only the affine part of this transformation on the
    given array of values.
    ``transform(values)`` is always equivalent to
    ``transform_affine(transform_non_affine(values))``.
    Keywords: matplotlib.transforms.Affine2DBase, transform_affine
    """
    output_var = input_var.transform_affine()

def matplotlib_transforms_Affine2DBase_transform_angles(self: 'any') -> 'any_computed':
    r"""
    Transform a set of angles anchored at specific locations.
    Parameters
    ----------
    angles : (N,) array-like
    Keywords: matplotlib.transforms.Affine2DBase, transform_angles
    """
    output_var = input_var.transform_angles()

def matplotlib_transforms_Affine2DBase_transform_bbox(self: 'any') -> 'any_computed':
    r"""
    Transform the given bounding box.
    For smarter transforms including caching (a common requirement in
    Matplotlib), see `TransformedBbox`.
    Keywords: matplotlib.transforms.Affine2DBase, transform_bbox
    """
    output_var = input_var.transform_bbox()

def matplotlib_transforms_Affine2DBase_transform_non_affine(self: 'any') -> 'any_computed':
    r"""
    Apply only the non-affine part of this transformation.
    ``transform(values)`` is always equivalent to
    ``transform_affine(transform_non_affine(values))``.
    Keywords: matplotlib.transforms.Affine2DBase, transform_non_affine
    """
    output_var = input_var.transform_non_affine()

def matplotlib_transforms_Affine2DBase_transform_path(self: 'any') -> 'any_computed':
    r"""
    Apply the transform to `.Path` *path*, returning a new `.Path`.
    In some cases, this transform may insert curves into the path
    that began as line segments.
    Keywords: matplotlib.transforms.Affine2DBase, transform_path
    """
    output_var = input_var.transform_path()

def matplotlib_transforms_Affine2DBase_transform_path_affine(self: 'any') -> 'any_computed':
    r"""
    Apply the affine part of this transform to `.Path` *path*, returning a
    new `.Path`.
    ``transform_path(path)`` is equivalent to
    ``transform_path_affine(transform_path_non_affine(values))``.
    Keywords: matplotlib.transforms.Affine2DBase, transform_path_affine
    """
    output_var = input_var.transform_path_affine()

def matplotlib_transforms_Affine2DBase_transform_path_non_affine(self: 'any') -> 'any_computed':
    r"""
    Apply the non-affine part of this transform to `.Path` *path*,
    returning a new `.Path`.
    ``transform_path(path)`` is equivalent to
    ``transform_path_affine(transform_path_non_affine(values))``.
    Keywords: matplotlib.transforms.Affine2DBase, transform_path_non_affine
    """
    output_var = input_var.transform_path_non_affine()

def matplotlib_transforms_Affine2DBase_transform_point(self: 'any') -> 'any_computed':
    r"""
    Return a transformed point.
    This function is only kept for backcompatibility; the more general
    `.transform` method is capable of transforming both a list of points
    and a single point.
    Keywords: matplotlib.transforms.Affine2DBase, transform_point
    """
    output_var = input_var.transform_point()

def matplotlib_transforms_AffineBase_contains_branch(self: 'any') -> 'any_computed':
    r"""
    Return whether the given transform is a sub-tree of this transform.
    This routine uses transform equality to identify sub-trees, therefore
    in many situations it is object id which will be used.
    Keywords: matplotlib.transforms.AffineBase, contains_branch
    """
    output_var = input_var.contains_branch()

def matplotlib_transforms_AffineBase_contains_branch_seperately(self: 'any') -> 'any_computed':
    r"""
    Return whether the given branch is a sub-tree of this transform on
    each separate dimension.
    A common use for this method is to identify if a transform is a blended
    transform containing an Axes' data transform. e.g.::
    Keywords: matplotlib.transforms.AffineBase, contains_branch_seperately
    """
    output_var = input_var.contains_branch_seperately()

def matplotlib_transforms_AffineBase_frozen(self: 'any') -> 'any_computed':
    r"""
    Return a frozen copy of this transform node.  The frozen copy will not
    be updated when its children change.  Useful for storing a previously
    known state of a transform where ``copy.deepcopy()`` might normally be
    used.
    Keywords: matplotlib.transforms.AffineBase, frozen
    """
    output_var = input_var.frozen()

def matplotlib_transforms_AffineBase_get_affine(self: 'any') -> 'any_computed':
    r"""
    Get the affine part of this transform.
    Keywords: matplotlib.transforms.AffineBase, get_affine
    """
    output_var = input_var.get_affine()

def matplotlib_transforms_AffineBase_get_matrix(self: 'any') -> 'any_computed':
    r"""
    Get the matrix for the affine part of this transform.
    Keywords: matplotlib.transforms.AffineBase, get_matrix
    """
    output_var = input_var.get_matrix()

def matplotlib_transforms_AffineBase_invalidate(self: 'any') -> 'any_computed':
    r"""
    Invalidate this `TransformNode` and triggers an invalidation of its
    ancestors.  Should be called any time the transform changes.
    Keywords: matplotlib.transforms.AffineBase, invalidate
    """
    output_var = input_var.invalidate()

def matplotlib_transforms_AffineBase_inverted(self: 'any') -> 'any_computed':
    r"""
    Return the corresponding inverse transformation.
    It holds ``x == self.inverted().transform(self.transform(x))``.
    The return value of this method should be treated as
    Keywords: matplotlib.transforms.AffineBase, inverted
    """
    output_var = input_var.inverted()

def matplotlib_transforms_AffineBase_set_children(self: 'any') -> 'any_computed':
    r"""
    Set the children of the transform, to let the invalidation
    system know which transforms can invalidate this transform.
    Should be called from the constructor of any transforms that
    depend on other transforms.
    Keywords: matplotlib.transforms.AffineBase, set_children
    """
    output_var = input_var.set_children()

def matplotlib_transforms_AffineBase_transform(self: 'any') -> 'any_computed':
    r"""
    Apply this transformation on the given array of *values*.
    Parameters
    ----------
    values : array-like
    Keywords: matplotlib.transforms.AffineBase, transform
    """
    output_var = input_var.transform()

def matplotlib_transforms_AffineBase_transform_affine(self: 'any') -> 'any_computed':
    r"""
    Apply only the affine part of this transformation on the
    given array of values.
    ``transform(values)`` is always equivalent to
    ``transform_affine(transform_non_affine(values))``.
    Keywords: matplotlib.transforms.AffineBase, transform_affine
    """
    output_var = input_var.transform_affine()

def matplotlib_transforms_AffineBase_transform_angles(self: 'any') -> 'any_computed':
    r"""
    Transform a set of angles anchored at specific locations.
    Parameters
    ----------
    angles : (N,) array-like
    Keywords: matplotlib.transforms.AffineBase, transform_angles
    """
    output_var = input_var.transform_angles()

def matplotlib_transforms_AffineBase_transform_bbox(self: 'any') -> 'any_computed':
    r"""
    Transform the given bounding box.
    For smarter transforms including caching (a common requirement in
    Matplotlib), see `TransformedBbox`.
    Keywords: matplotlib.transforms.AffineBase, transform_bbox
    """
    output_var = input_var.transform_bbox()

def matplotlib_transforms_AffineBase_transform_non_affine(self: 'any') -> 'any_computed':
    r"""
    Apply only the non-affine part of this transformation.
    ``transform(values)`` is always equivalent to
    ``transform_affine(transform_non_affine(values))``.
    Keywords: matplotlib.transforms.AffineBase, transform_non_affine
    """
    output_var = input_var.transform_non_affine()

def matplotlib_transforms_AffineBase_transform_path(self: 'any') -> 'any_computed':
    r"""
    Apply the transform to `.Path` *path*, returning a new `.Path`.
    In some cases, this transform may insert curves into the path
    that began as line segments.
    Keywords: matplotlib.transforms.AffineBase, transform_path
    """
    output_var = input_var.transform_path()

def matplotlib_transforms_AffineBase_transform_path_affine(self: 'any') -> 'any_computed':
    r"""
    Apply the affine part of this transform to `.Path` *path*, returning a
    new `.Path`.
    ``transform_path(path)`` is equivalent to
    ``transform_path_affine(transform_path_non_affine(values))``.
    Keywords: matplotlib.transforms.AffineBase, transform_path_affine
    """
    output_var = input_var.transform_path_affine()

def matplotlib_transforms_AffineBase_transform_path_non_affine(self: 'any') -> 'any_computed':
    r"""
    Apply the non-affine part of this transform to `.Path` *path*,
    returning a new `.Path`.
    ``transform_path(path)`` is equivalent to
    ``transform_path_affine(transform_path_non_affine(values))``.
    Keywords: matplotlib.transforms.AffineBase, transform_path_non_affine
    """
    output_var = input_var.transform_path_non_affine()

def matplotlib_transforms_AffineBase_transform_point(self: 'any') -> 'any_computed':
    r"""
    Return a transformed point.
    This function is only kept for backcompatibility; the more general
    `.transform` method is capable of transforming both a list of points
    and a single point.
    Keywords: matplotlib.transforms.AffineBase, transform_point
    """
    output_var = input_var.transform_point()

def matplotlib_transforms_AffineDeltaTransform_contains_branch(self: 'any') -> 'any_computed':
    r"""
    Return whether the given transform is a sub-tree of this transform.
    This routine uses transform equality to identify sub-trees, therefore
    in many situations it is object id which will be used.
    Keywords: matplotlib.transforms.AffineDeltaTransform, contains_branch
    """
    output_var = input_var.contains_branch()

def matplotlib_transforms_AffineDeltaTransform_contains_branch_seperately(self: 'any') -> 'any_computed':
    r"""
    Return whether the given branch is a sub-tree of this transform on
    each separate dimension.
    A common use for this method is to identify if a transform is a blended
    transform containing an Axes' data transform. e.g.::
    Keywords: matplotlib.transforms.AffineDeltaTransform, contains_branch_seperately
    """
    output_var = input_var.contains_branch_seperately()

def matplotlib_transforms_AffineDeltaTransform_frozen(self: 'any') -> 'any_computed':
    r"""
    Return a frozen copy of this transform node.  The frozen copy will not
    be updated when its children change.  Useful for storing a previously
    known state of a transform where ``copy.deepcopy()`` might normally be
    used.
    Keywords: matplotlib.transforms.AffineDeltaTransform, frozen
    """
    output_var = input_var.frozen()

def matplotlib_transforms_AffineDeltaTransform_get_affine(self: 'any') -> 'any_computed':
    r"""
    Get the affine part of this transform.
    Keywords: matplotlib.transforms.AffineDeltaTransform, get_affine
    """
    output_var = input_var.get_affine()

def matplotlib_transforms_AffineDeltaTransform_get_matrix(self: 'any') -> 'any_computed':
    r"""
    Get the matrix for the affine part of this transform.
    Keywords: matplotlib.transforms.AffineDeltaTransform, get_matrix
    """
    output_var = input_var.get_matrix()

def matplotlib_transforms_AffineDeltaTransform_invalidate(self: 'any') -> 'any_computed':
    r"""
    Invalidate this `TransformNode` and triggers an invalidation of its
    ancestors.  Should be called any time the transform changes.
    Keywords: matplotlib.transforms.AffineDeltaTransform, invalidate
    """
    output_var = input_var.invalidate()

def matplotlib_transforms_AffineDeltaTransform_inverted(self: 'any') -> 'any_computed':
    r"""
    Return the corresponding inverse transformation.
    It holds ``x == self.inverted().transform(self.transform(x))``.
    The return value of this method should be treated as
    Keywords: matplotlib.transforms.AffineDeltaTransform, inverted
    """
    output_var = input_var.inverted()

def matplotlib_transforms_AffineDeltaTransform_set_children(self: 'any') -> 'any_computed':
    r"""
    Set the children of the transform, to let the invalidation
    system know which transforms can invalidate this transform.
    Should be called from the constructor of any transforms that
    depend on other transforms.
    Keywords: matplotlib.transforms.AffineDeltaTransform, set_children
    """
    output_var = input_var.set_children()

def matplotlib_transforms_AffineDeltaTransform_to_values(self: 'any') -> 'any_computed':
    r"""
    Return the values of the matrix as an ``(a, b, c, d, e, f)`` tuple.
    Keywords: matplotlib.transforms.AffineDeltaTransform, to_values
    """
    output_var = input_var.to_values()

def matplotlib_transforms_AffineDeltaTransform_transform(self: 'any') -> 'any_computed':
    r"""
    Apply this transformation on the given array of *values*.
    Parameters
    ----------
    values : array-like
    Keywords: matplotlib.transforms.AffineDeltaTransform, transform
    """
    output_var = input_var.transform()

def matplotlib_transforms_AffineDeltaTransform_transform_affine(self: 'any') -> 'any_computed':
    r"""
    Apply only the affine part of this transformation on the
    given array of values.
    ``transform(values)`` is always equivalent to
    ``transform_affine(transform_non_affine(values))``.
    Keywords: matplotlib.transforms.AffineDeltaTransform, transform_affine
    """
    output_var = input_var.transform_affine()

def matplotlib_transforms_AffineDeltaTransform_transform_angles(self: 'any') -> 'any_computed':
    r"""
    Transform a set of angles anchored at specific locations.
    Parameters
    ----------
    angles : (N,) array-like
    Keywords: matplotlib.transforms.AffineDeltaTransform, transform_angles
    """
    output_var = input_var.transform_angles()

def matplotlib_transforms_AffineDeltaTransform_transform_bbox(self: 'any') -> 'any_computed':
    r"""
    Transform the given bounding box.
    For smarter transforms including caching (a common requirement in
    Matplotlib), see `TransformedBbox`.
    Keywords: matplotlib.transforms.AffineDeltaTransform, transform_bbox
    """
    output_var = input_var.transform_bbox()

def matplotlib_transforms_AffineDeltaTransform_transform_non_affine(self: 'any') -> 'any_computed':
    r"""
    Apply only the non-affine part of this transformation.
    ``transform(values)`` is always equivalent to
    ``transform_affine(transform_non_affine(values))``.
    Keywords: matplotlib.transforms.AffineDeltaTransform, transform_non_affine
    """
    output_var = input_var.transform_non_affine()

def matplotlib_transforms_AffineDeltaTransform_transform_path(self: 'any') -> 'any_computed':
    r"""
    Apply the transform to `.Path` *path*, returning a new `.Path`.
    In some cases, this transform may insert curves into the path
    that began as line segments.
    Keywords: matplotlib.transforms.AffineDeltaTransform, transform_path
    """
    output_var = input_var.transform_path()

def matplotlib_transforms_AffineDeltaTransform_transform_path_affine(self: 'any') -> 'any_computed':
    r"""
    Apply the affine part of this transform to `.Path` *path*, returning a
    new `.Path`.
    ``transform_path(path)`` is equivalent to
    ``transform_path_affine(transform_path_non_affine(values))``.
    Keywords: matplotlib.transforms.AffineDeltaTransform, transform_path_affine
    """
    output_var = input_var.transform_path_affine()

def matplotlib_transforms_AffineDeltaTransform_transform_path_non_affine(self: 'any') -> 'any_computed':
    r"""
    Apply the non-affine part of this transform to `.Path` *path*,
    returning a new `.Path`.
    ``transform_path(path)`` is equivalent to
    ``transform_path_affine(transform_path_non_affine(values))``.
    Keywords: matplotlib.transforms.AffineDeltaTransform, transform_path_non_affine
    """
    output_var = input_var.transform_path_non_affine()

def matplotlib_transforms_AffineDeltaTransform_transform_point(self: 'any') -> 'any_computed':
    r"""
    Return a transformed point.
    This function is only kept for backcompatibility; the more general
    `.transform` method is capable of transforming both a list of points
    and a single point.
    Keywords: matplotlib.transforms.AffineDeltaTransform, transform_point
    """
    output_var = input_var.transform_point()

def matplotlib_transforms_BboxTransform_contains_branch(self: 'any') -> 'any_computed':
    r"""
    Return whether the given transform is a sub-tree of this transform.
    This routine uses transform equality to identify sub-trees, therefore
    in many situations it is object id which will be used.
    Keywords: matplotlib.transforms.BboxTransform, contains_branch
    """
    output_var = input_var.contains_branch()

def matplotlib_transforms_BboxTransform_contains_branch_seperately(self: 'any') -> 'any_computed':
    r"""
    Return whether the given branch is a sub-tree of this transform on
    each separate dimension.
    A common use for this method is to identify if a transform is a blended
    transform containing an Axes' data transform. e.g.::
    Keywords: matplotlib.transforms.BboxTransform, contains_branch_seperately
    """
    output_var = input_var.contains_branch_seperately()

def matplotlib_transforms_BboxTransform_frozen(self: 'any') -> 'any_computed':
    r"""
    Return a frozen copy of this transform node.  The frozen copy will not
    be updated when its children change.  Useful for storing a previously
    known state of a transform where ``copy.deepcopy()`` might normally be
    used.
    Keywords: matplotlib.transforms.BboxTransform, frozen
    """
    output_var = input_var.frozen()

def matplotlib_transforms_BboxTransform_get_affine(self: 'any') -> 'any_computed':
    r"""
    Get the affine part of this transform.
    Keywords: matplotlib.transforms.BboxTransform, get_affine
    """
    output_var = input_var.get_affine()

def matplotlib_transforms_BboxTransform_get_matrix(self: 'any') -> 'any_computed':
    r"""
    Get the matrix for the affine part of this transform.
    Keywords: matplotlib.transforms.BboxTransform, get_matrix
    """
    output_var = input_var.get_matrix()

def matplotlib_transforms_BboxTransform_invalidate(self: 'any') -> 'any_computed':
    r"""
    Invalidate this `TransformNode` and triggers an invalidation of its
    ancestors.  Should be called any time the transform changes.
    Keywords: matplotlib.transforms.BboxTransform, invalidate
    """
    output_var = input_var.invalidate()

def matplotlib_transforms_BboxTransform_inverted(self: 'any') -> 'any_computed':
    r"""
    Return the corresponding inverse transformation.
    It holds ``x == self.inverted().transform(self.transform(x))``.
    The return value of this method should be treated as
    Keywords: matplotlib.transforms.BboxTransform, inverted
    """
    output_var = input_var.inverted()

def matplotlib_transforms_BboxTransform_set_children(self: 'any') -> 'any_computed':
    r"""
    Set the children of the transform, to let the invalidation
    system know which transforms can invalidate this transform.
    Should be called from the constructor of any transforms that
    depend on other transforms.
    Keywords: matplotlib.transforms.BboxTransform, set_children
    """
    output_var = input_var.set_children()

def matplotlib_transforms_BboxTransform_to_values(self: 'any') -> 'any_computed':
    r"""
    Return the values of the matrix as an ``(a, b, c, d, e, f)`` tuple.
    Keywords: matplotlib.transforms.BboxTransform, to_values
    """
    output_var = input_var.to_values()

def matplotlib_transforms_BboxTransform_transform(self: 'any') -> 'any_computed':
    r"""
    Apply this transformation on the given array of *values*.
    Parameters
    ----------
    values : array-like
    Keywords: matplotlib.transforms.BboxTransform, transform
    """
    output_var = input_var.transform()

def matplotlib_transforms_BboxTransform_transform_affine(self: 'any') -> 'any_computed':
    r"""
    Apply only the affine part of this transformation on the
    given array of values.
    ``transform(values)`` is always equivalent to
    ``transform_affine(transform_non_affine(values))``.
    Keywords: matplotlib.transforms.BboxTransform, transform_affine
    """
    output_var = input_var.transform_affine()

def matplotlib_transforms_BboxTransform_transform_angles(self: 'any') -> 'any_computed':
    r"""
    Transform a set of angles anchored at specific locations.
    Parameters
    ----------
    angles : (N,) array-like
    Keywords: matplotlib.transforms.BboxTransform, transform_angles
    """
    output_var = input_var.transform_angles()

def matplotlib_transforms_BboxTransform_transform_bbox(self: 'any') -> 'any_computed':
    r"""
    Transform the given bounding box.
    For smarter transforms including caching (a common requirement in
    Matplotlib), see `TransformedBbox`.
    Keywords: matplotlib.transforms.BboxTransform, transform_bbox
    """
    output_var = input_var.transform_bbox()

def matplotlib_transforms_BboxTransform_transform_non_affine(self: 'any') -> 'any_computed':
    r"""
    Apply only the non-affine part of this transformation.
    ``transform(values)`` is always equivalent to
    ``transform_affine(transform_non_affine(values))``.
    Keywords: matplotlib.transforms.BboxTransform, transform_non_affine
    """
    output_var = input_var.transform_non_affine()

def matplotlib_transforms_BboxTransform_transform_path(self: 'any') -> 'any_computed':
    r"""
    Apply the transform to `.Path` *path*, returning a new `.Path`.
    In some cases, this transform may insert curves into the path
    that began as line segments.
    Keywords: matplotlib.transforms.BboxTransform, transform_path
    """
    output_var = input_var.transform_path()

def matplotlib_transforms_BboxTransform_transform_path_affine(self: 'any') -> 'any_computed':
    r"""
    Apply the affine part of this transform to `.Path` *path*, returning a
    new `.Path`.
    ``transform_path(path)`` is equivalent to
    ``transform_path_affine(transform_path_non_affine(values))``.
    Keywords: matplotlib.transforms.BboxTransform, transform_path_affine
    """
    output_var = input_var.transform_path_affine()

def matplotlib_transforms_BboxTransform_transform_path_non_affine(self: 'any') -> 'any_computed':
    r"""
    Apply the non-affine part of this transform to `.Path` *path*,
    returning a new `.Path`.
    ``transform_path(path)`` is equivalent to
    ``transform_path_affine(transform_path_non_affine(values))``.
    Keywords: matplotlib.transforms.BboxTransform, transform_path_non_affine
    """
    output_var = input_var.transform_path_non_affine()

def matplotlib_transforms_BboxTransform_transform_point(self: 'any') -> 'any_computed':
    r"""
    Return a transformed point.
    This function is only kept for backcompatibility; the more general
    `.transform` method is capable of transforming both a list of points
    and a single point.
    Keywords: matplotlib.transforms.BboxTransform, transform_point
    """
    output_var = input_var.transform_point()

def matplotlib_transforms_BboxTransformFrom_contains_branch(self: 'any') -> 'any_computed':
    r"""
    Return whether the given transform is a sub-tree of this transform.
    This routine uses transform equality to identify sub-trees, therefore
    in many situations it is object id which will be used.
    Keywords: matplotlib.transforms.BboxTransformFrom, contains_branch
    """
    output_var = input_var.contains_branch()

def matplotlib_transforms_BboxTransformFrom_contains_branch_seperately(self: 'any') -> 'any_computed':
    r"""
    Return whether the given branch is a sub-tree of this transform on
    each separate dimension.
    A common use for this method is to identify if a transform is a blended
    transform containing an Axes' data transform. e.g.::
    Keywords: matplotlib.transforms.BboxTransformFrom, contains_branch_seperately
    """
    output_var = input_var.contains_branch_seperately()

def matplotlib_transforms_BboxTransformFrom_frozen(self: 'any') -> 'any_computed':
    r"""
    Return a frozen copy of this transform node.  The frozen copy will not
    be updated when its children change.  Useful for storing a previously
    known state of a transform where ``copy.deepcopy()`` might normally be
    used.
    Keywords: matplotlib.transforms.BboxTransformFrom, frozen
    """
    output_var = input_var.frozen()

def matplotlib_transforms_BboxTransformFrom_get_affine(self: 'any') -> 'any_computed':
    r"""
    Get the affine part of this transform.
    Keywords: matplotlib.transforms.BboxTransformFrom, get_affine
    """
    output_var = input_var.get_affine()

def matplotlib_transforms_BboxTransformFrom_get_matrix(self: 'any') -> 'any_computed':
    r"""
    Get the matrix for the affine part of this transform.
    Keywords: matplotlib.transforms.BboxTransformFrom, get_matrix
    """
    output_var = input_var.get_matrix()

def matplotlib_transforms_BboxTransformFrom_invalidate(self: 'any') -> 'any_computed':
    r"""
    Invalidate this `TransformNode` and triggers an invalidation of its
    ancestors.  Should be called any time the transform changes.
    Keywords: matplotlib.transforms.BboxTransformFrom, invalidate
    """
    output_var = input_var.invalidate()

def matplotlib_transforms_BboxTransformFrom_inverted(self: 'any') -> 'any_computed':
    r"""
    Return the corresponding inverse transformation.
    It holds ``x == self.inverted().transform(self.transform(x))``.
    The return value of this method should be treated as
    Keywords: matplotlib.transforms.BboxTransformFrom, inverted
    """
    output_var = input_var.inverted()

def matplotlib_transforms_BboxTransformFrom_set_children(self: 'any') -> 'any_computed':
    r"""
    Set the children of the transform, to let the invalidation
    system know which transforms can invalidate this transform.
    Should be called from the constructor of any transforms that
    depend on other transforms.
    Keywords: matplotlib.transforms.BboxTransformFrom, set_children
    """
    output_var = input_var.set_children()

def matplotlib_transforms_BboxTransformFrom_to_values(self: 'any') -> 'any_computed':
    r"""
    Return the values of the matrix as an ``(a, b, c, d, e, f)`` tuple.
    Keywords: matplotlib.transforms.BboxTransformFrom, to_values
    """
    output_var = input_var.to_values()

def matplotlib_transforms_BboxTransformFrom_transform(self: 'any') -> 'any_computed':
    r"""
    Apply this transformation on the given array of *values*.
    Parameters
    ----------
    values : array-like
    Keywords: matplotlib.transforms.BboxTransformFrom, transform
    """
    output_var = input_var.transform()

def matplotlib_transforms_BboxTransformFrom_transform_affine(self: 'any') -> 'any_computed':
    r"""
    Apply only the affine part of this transformation on the
    given array of values.
    ``transform(values)`` is always equivalent to
    ``transform_affine(transform_non_affine(values))``.
    Keywords: matplotlib.transforms.BboxTransformFrom, transform_affine
    """
    output_var = input_var.transform_affine()

def matplotlib_transforms_BboxTransformFrom_transform_angles(self: 'any') -> 'any_computed':
    r"""
    Transform a set of angles anchored at specific locations.
    Parameters
    ----------
    angles : (N,) array-like
    Keywords: matplotlib.transforms.BboxTransformFrom, transform_angles
    """
    output_var = input_var.transform_angles()

def matplotlib_transforms_BboxTransformFrom_transform_bbox(self: 'any') -> 'any_computed':
    r"""
    Transform the given bounding box.
    For smarter transforms including caching (a common requirement in
    Matplotlib), see `TransformedBbox`.
    Keywords: matplotlib.transforms.BboxTransformFrom, transform_bbox
    """
    output_var = input_var.transform_bbox()

def matplotlib_transforms_BboxTransformFrom_transform_non_affine(self: 'any') -> 'any_computed':
    r"""
    Apply only the non-affine part of this transformation.
    ``transform(values)`` is always equivalent to
    ``transform_affine(transform_non_affine(values))``.
    Keywords: matplotlib.transforms.BboxTransformFrom, transform_non_affine
    """
    output_var = input_var.transform_non_affine()

def matplotlib_transforms_BboxTransformFrom_transform_path(self: 'any') -> 'any_computed':
    r"""
    Apply the transform to `.Path` *path*, returning a new `.Path`.
    In some cases, this transform may insert curves into the path
    that began as line segments.
    Keywords: matplotlib.transforms.BboxTransformFrom, transform_path
    """
    output_var = input_var.transform_path()

def matplotlib_transforms_BboxTransformFrom_transform_path_affine(self: 'any') -> 'any_computed':
    r"""
    Apply the affine part of this transform to `.Path` *path*, returning a
    new `.Path`.
    ``transform_path(path)`` is equivalent to
    ``transform_path_affine(transform_path_non_affine(values))``.
    Keywords: matplotlib.transforms.BboxTransformFrom, transform_path_affine
    """
    output_var = input_var.transform_path_affine()

def matplotlib_transforms_BboxTransformFrom_transform_path_non_affine(self: 'any') -> 'any_computed':
    r"""
    Apply the non-affine part of this transform to `.Path` *path*,
    returning a new `.Path`.
    ``transform_path(path)`` is equivalent to
    ``transform_path_affine(transform_path_non_affine(values))``.
    Keywords: matplotlib.transforms.BboxTransformFrom, transform_path_non_affine
    """
    output_var = input_var.transform_path_non_affine()

def matplotlib_transforms_BboxTransformFrom_transform_point(self: 'any') -> 'any_computed':
    r"""
    Return a transformed point.
    This function is only kept for backcompatibility; the more general
    `.transform` method is capable of transforming both a list of points
    and a single point.
    Keywords: matplotlib.transforms.BboxTransformFrom, transform_point
    """
    output_var = input_var.transform_point()

def matplotlib_transforms_BboxTransformTo_contains_branch(self: 'any') -> 'any_computed':
    r"""
    Return whether the given transform is a sub-tree of this transform.
    This routine uses transform equality to identify sub-trees, therefore
    in many situations it is object id which will be used.
    Keywords: matplotlib.transforms.BboxTransformTo, contains_branch
    """
    output_var = input_var.contains_branch()

def matplotlib_transforms_BboxTransformTo_contains_branch_seperately(self: 'any') -> 'any_computed':
    r"""
    Return whether the given branch is a sub-tree of this transform on
    each separate dimension.
    A common use for this method is to identify if a transform is a blended
    transform containing an Axes' data transform. e.g.::
    Keywords: matplotlib.transforms.BboxTransformTo, contains_branch_seperately
    """
    output_var = input_var.contains_branch_seperately()

def matplotlib_transforms_BboxTransformTo_frozen(self: 'any') -> 'any_computed':
    r"""
    Return a frozen copy of this transform node.  The frozen copy will not
    be updated when its children change.  Useful for storing a previously
    known state of a transform where ``copy.deepcopy()`` might normally be
    used.
    Keywords: matplotlib.transforms.BboxTransformTo, frozen
    """
    output_var = input_var.frozen()

def matplotlib_transforms_BboxTransformTo_get_affine(self: 'any') -> 'any_computed':
    r"""
    Get the affine part of this transform.
    Keywords: matplotlib.transforms.BboxTransformTo, get_affine
    """
    output_var = input_var.get_affine()

def matplotlib_transforms_BboxTransformTo_get_matrix(self: 'any') -> 'any_computed':
    r"""
    Get the matrix for the affine part of this transform.
    Keywords: matplotlib.transforms.BboxTransformTo, get_matrix
    """
    output_var = input_var.get_matrix()

def matplotlib_transforms_BboxTransformTo_invalidate(self: 'any') -> 'any_computed':
    r"""
    Invalidate this `TransformNode` and triggers an invalidation of its
    ancestors.  Should be called any time the transform changes.
    Keywords: matplotlib.transforms.BboxTransformTo, invalidate
    """
    output_var = input_var.invalidate()

def matplotlib_transforms_BboxTransformTo_inverted(self: 'any') -> 'any_computed':
    r"""
    Return the corresponding inverse transformation.
    It holds ``x == self.inverted().transform(self.transform(x))``.
    The return value of this method should be treated as
    Keywords: matplotlib.transforms.BboxTransformTo, inverted
    """
    output_var = input_var.inverted()

def matplotlib_transforms_BboxTransformTo_set_children(self: 'any') -> 'any_computed':
    r"""
    Set the children of the transform, to let the invalidation
    system know which transforms can invalidate this transform.
    Should be called from the constructor of any transforms that
    depend on other transforms.
    Keywords: matplotlib.transforms.BboxTransformTo, set_children
    """
    output_var = input_var.set_children()

def matplotlib_transforms_BboxTransformTo_to_values(self: 'any') -> 'any_computed':
    r"""
    Return the values of the matrix as an ``(a, b, c, d, e, f)`` tuple.
    Keywords: matplotlib.transforms.BboxTransformTo, to_values
    """
    output_var = input_var.to_values()

def matplotlib_transforms_BboxTransformTo_transform(self: 'any') -> 'any_computed':
    r"""
    Apply this transformation on the given array of *values*.
    Parameters
    ----------
    values : array-like
    Keywords: matplotlib.transforms.BboxTransformTo, transform
    """
    output_var = input_var.transform()

def matplotlib_transforms_BboxTransformTo_transform_affine(self: 'any') -> 'any_computed':
    r"""
    Apply only the affine part of this transformation on the
    given array of values.
    ``transform(values)`` is always equivalent to
    ``transform_affine(transform_non_affine(values))``.
    Keywords: matplotlib.transforms.BboxTransformTo, transform_affine
    """
    output_var = input_var.transform_affine()

def matplotlib_transforms_BboxTransformTo_transform_angles(self: 'any') -> 'any_computed':
    r"""
    Transform a set of angles anchored at specific locations.
    Parameters
    ----------
    angles : (N,) array-like
    Keywords: matplotlib.transforms.BboxTransformTo, transform_angles
    """
    output_var = input_var.transform_angles()

def matplotlib_transforms_BboxTransformTo_transform_bbox(self: 'any') -> 'any_computed':
    r"""
    Transform the given bounding box.
    For smarter transforms including caching (a common requirement in
    Matplotlib), see `TransformedBbox`.
    Keywords: matplotlib.transforms.BboxTransformTo, transform_bbox
    """
    output_var = input_var.transform_bbox()

def matplotlib_transforms_BboxTransformTo_transform_non_affine(self: 'any') -> 'any_computed':
    r"""
    Apply only the non-affine part of this transformation.
    ``transform(values)`` is always equivalent to
    ``transform_affine(transform_non_affine(values))``.
    Keywords: matplotlib.transforms.BboxTransformTo, transform_non_affine
    """
    output_var = input_var.transform_non_affine()

def matplotlib_transforms_BboxTransformTo_transform_path(self: 'any') -> 'any_computed':
    r"""
    Apply the transform to `.Path` *path*, returning a new `.Path`.
    In some cases, this transform may insert curves into the path
    that began as line segments.
    Keywords: matplotlib.transforms.BboxTransformTo, transform_path
    """
    output_var = input_var.transform_path()

def matplotlib_transforms_BboxTransformTo_transform_path_affine(self: 'any') -> 'any_computed':
    r"""
    Apply the affine part of this transform to `.Path` *path*, returning a
    new `.Path`.
    ``transform_path(path)`` is equivalent to
    ``transform_path_affine(transform_path_non_affine(values))``.
    Keywords: matplotlib.transforms.BboxTransformTo, transform_path_affine
    """
    output_var = input_var.transform_path_affine()

def matplotlib_transforms_BboxTransformTo_transform_path_non_affine(self: 'any') -> 'any_computed':
    r"""
    Apply the non-affine part of this transform to `.Path` *path*,
    returning a new `.Path`.
    ``transform_path(path)`` is equivalent to
    ``transform_path_affine(transform_path_non_affine(values))``.
    Keywords: matplotlib.transforms.BboxTransformTo, transform_path_non_affine
    """
    output_var = input_var.transform_path_non_affine()

def matplotlib_transforms_BboxTransformTo_transform_point(self: 'any') -> 'any_computed':
    r"""
    Return a transformed point.
    This function is only kept for backcompatibility; the more general
    `.transform` method is capable of transforming both a list of points
    and a single point.
    Keywords: matplotlib.transforms.BboxTransformTo, transform_point
    """
    output_var = input_var.transform_point()

def matplotlib_transforms_BboxTransformToMaxOnly_contains_branch(self: 'any') -> 'any_computed':
    r"""
    Return whether the given transform is a sub-tree of this transform.
    This routine uses transform equality to identify sub-trees, therefore
    in many situations it is object id which will be used.
    Keywords: matplotlib.transforms.BboxTransformToMaxOnly, contains_branch
    """
    output_var = input_var.contains_branch()

def matplotlib_transforms_BboxTransformToMaxOnly_contains_branch_seperately(self: 'any') -> 'any_computed':
    r"""
    Return whether the given branch is a sub-tree of this transform on
    each separate dimension.
    A common use for this method is to identify if a transform is a blended
    transform containing an Axes' data transform. e.g.::
    Keywords: matplotlib.transforms.BboxTransformToMaxOnly, contains_branch_seperately
    """
    output_var = input_var.contains_branch_seperately()

def matplotlib_transforms_BboxTransformToMaxOnly_frozen(self: 'any') -> 'any_computed':
    r"""
    Return a frozen copy of this transform node.  The frozen copy will not
    be updated when its children change.  Useful for storing a previously
    known state of a transform where ``copy.deepcopy()`` might normally be
    used.
    Keywords: matplotlib.transforms.BboxTransformToMaxOnly, frozen
    """
    output_var = input_var.frozen()

def matplotlib_transforms_BboxTransformToMaxOnly_get_affine(self: 'any') -> 'any_computed':
    r"""
    Get the affine part of this transform.
    Keywords: matplotlib.transforms.BboxTransformToMaxOnly, get_affine
    """
    output_var = input_var.get_affine()

def matplotlib_transforms_BboxTransformToMaxOnly_get_matrix(self: 'any') -> 'any_computed':
    r"""
    Get the matrix for the affine part of this transform.
    Keywords: matplotlib.transforms.BboxTransformToMaxOnly, get_matrix
    """
    output_var = input_var.get_matrix()

def matplotlib_transforms_BboxTransformToMaxOnly_invalidate(self: 'any') -> 'any_computed':
    r"""
    Invalidate this `TransformNode` and triggers an invalidation of its
    ancestors.  Should be called any time the transform changes.
    Keywords: matplotlib.transforms.BboxTransformToMaxOnly, invalidate
    """
    output_var = input_var.invalidate()

def matplotlib_transforms_BboxTransformToMaxOnly_inverted(self: 'any') -> 'any_computed':
    r"""
    Return the corresponding inverse transformation.
    It holds ``x == self.inverted().transform(self.transform(x))``.
    The return value of this method should be treated as
    Keywords: matplotlib.transforms.BboxTransformToMaxOnly, inverted
    """
    output_var = input_var.inverted()

def matplotlib_transforms_BboxTransformToMaxOnly_set_children(self: 'any') -> 'any_computed':
    r"""
    Set the children of the transform, to let the invalidation
    system know which transforms can invalidate this transform.
    Should be called from the constructor of any transforms that
    depend on other transforms.
    Keywords: matplotlib.transforms.BboxTransformToMaxOnly, set_children
    """
    output_var = input_var.set_children()

def matplotlib_transforms_BboxTransformToMaxOnly_to_values(self: 'any') -> 'any_computed':
    r"""
    Return the values of the matrix as an ``(a, b, c, d, e, f)`` tuple.
    Keywords: matplotlib.transforms.BboxTransformToMaxOnly, to_values
    """
    output_var = input_var.to_values()

def matplotlib_transforms_BboxTransformToMaxOnly_transform(self: 'any') -> 'any_computed':
    r"""
    Apply this transformation on the given array of *values*.
    Parameters
    ----------
    values : array-like
    Keywords: matplotlib.transforms.BboxTransformToMaxOnly, transform
    """
    output_var = input_var.transform()

def matplotlib_transforms_BboxTransformToMaxOnly_transform_affine(self: 'any') -> 'any_computed':
    r"""
    Apply only the affine part of this transformation on the
    given array of values.
    ``transform(values)`` is always equivalent to
    ``transform_affine(transform_non_affine(values))``.
    Keywords: matplotlib.transforms.BboxTransformToMaxOnly, transform_affine
    """
    output_var = input_var.transform_affine()

def matplotlib_transforms_BboxTransformToMaxOnly_transform_angles(self: 'any') -> 'any_computed':
    r"""
    Transform a set of angles anchored at specific locations.
    Parameters
    ----------
    angles : (N,) array-like
    Keywords: matplotlib.transforms.BboxTransformToMaxOnly, transform_angles
    """
    output_var = input_var.transform_angles()

def matplotlib_transforms_BboxTransformToMaxOnly_transform_bbox(self: 'any') -> 'any_computed':
    r"""
    Transform the given bounding box.
    For smarter transforms including caching (a common requirement in
    Matplotlib), see `TransformedBbox`.
    Keywords: matplotlib.transforms.BboxTransformToMaxOnly, transform_bbox
    """
    output_var = input_var.transform_bbox()

def matplotlib_transforms_BboxTransformToMaxOnly_transform_non_affine(self: 'any') -> 'any_computed':
    r"""
    Apply only the non-affine part of this transformation.
    ``transform(values)`` is always equivalent to
    ``transform_affine(transform_non_affine(values))``.
    Keywords: matplotlib.transforms.BboxTransformToMaxOnly, transform_non_affine
    """
    output_var = input_var.transform_non_affine()

def matplotlib_transforms_BboxTransformToMaxOnly_transform_path(self: 'any') -> 'any_computed':
    r"""
    Apply the transform to `.Path` *path*, returning a new `.Path`.
    In some cases, this transform may insert curves into the path
    that began as line segments.
    Keywords: matplotlib.transforms.BboxTransformToMaxOnly, transform_path
    """
    output_var = input_var.transform_path()

def matplotlib_transforms_BboxTransformToMaxOnly_transform_path_affine(self: 'any') -> 'any_computed':
    r"""
    Apply the affine part of this transform to `.Path` *path*, returning a
    new `.Path`.
    ``transform_path(path)`` is equivalent to
    ``transform_path_affine(transform_path_non_affine(values))``.
    Keywords: matplotlib.transforms.BboxTransformToMaxOnly, transform_path_affine
    """
    output_var = input_var.transform_path_affine()

def matplotlib_transforms_BboxTransformToMaxOnly_transform_path_non_affine(self: 'any') -> 'any_computed':
    r"""
    Apply the non-affine part of this transform to `.Path` *path*,
    returning a new `.Path`.
    ``transform_path(path)`` is equivalent to
    ``transform_path_affine(transform_path_non_affine(values))``.
    Keywords: matplotlib.transforms.BboxTransformToMaxOnly, transform_path_non_affine
    """
    output_var = input_var.transform_path_non_affine()

def matplotlib_transforms_BboxTransformToMaxOnly_transform_point(self: 'any') -> 'any_computed':
    r"""
    Return a transformed point.
    This function is only kept for backcompatibility; the more general
    `.transform` method is capable of transforming both a list of points
    and a single point.
    Keywords: matplotlib.transforms.BboxTransformToMaxOnly, transform_point
    """
    output_var = input_var.transform_point()

def matplotlib_transforms_BlendedAffine2D_contains_branch(self: 'any') -> 'any_computed':
    r"""
    Return whether the given transform is a sub-tree of this transform.
    This routine uses transform equality to identify sub-trees, therefore
    in many situations it is object id which will be used.
    Keywords: matplotlib.transforms.BlendedAffine2D, contains_branch
    """
    output_var = input_var.contains_branch()

def matplotlib_transforms_BlendedAffine2D_contains_branch_seperately(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.transforms.BlendedAffine2D, contains_branch_seperately
    """
    output_var = input_var.contains_branch_seperately()

def matplotlib_transforms_BlendedAffine2D_frozen(self: 'any') -> 'any_computed':
    r"""
    Return a frozen copy of this transform node.  The frozen copy will not
    be updated when its children change.  Useful for storing a previously
    known state of a transform where ``copy.deepcopy()`` might normally be
    used.
    Keywords: matplotlib.transforms.BlendedAffine2D, frozen
    """
    output_var = input_var.frozen()

def matplotlib_transforms_BlendedAffine2D_get_affine(self: 'any') -> 'any_computed':
    r"""
    Get the affine part of this transform.
    Keywords: matplotlib.transforms.BlendedAffine2D, get_affine
    """
    output_var = input_var.get_affine()

def matplotlib_transforms_BlendedAffine2D_get_matrix(self: 'any') -> 'any_computed':
    r"""
    Get the matrix for the affine part of this transform.
    Keywords: matplotlib.transforms.BlendedAffine2D, get_matrix
    """
    output_var = input_var.get_matrix()

def matplotlib_transforms_BlendedAffine2D_invalidate(self: 'any') -> 'any_computed':
    r"""
    Invalidate this `TransformNode` and triggers an invalidation of its
    ancestors.  Should be called any time the transform changes.
    Keywords: matplotlib.transforms.BlendedAffine2D, invalidate
    """
    output_var = input_var.invalidate()

def matplotlib_transforms_BlendedAffine2D_inverted(self: 'any') -> 'any_computed':
    r"""
    Return the corresponding inverse transformation.
    It holds ``x == self.inverted().transform(self.transform(x))``.
    The return value of this method should be treated as
    Keywords: matplotlib.transforms.BlendedAffine2D, inverted
    """
    output_var = input_var.inverted()

def matplotlib_transforms_BlendedAffine2D_set_children(self: 'any') -> 'any_computed':
    r"""
    Set the children of the transform, to let the invalidation
    system know which transforms can invalidate this transform.
    Should be called from the constructor of any transforms that
    depend on other transforms.
    Keywords: matplotlib.transforms.BlendedAffine2D, set_children
    """
    output_var = input_var.set_children()

def matplotlib_transforms_BlendedAffine2D_to_values(self: 'any') -> 'any_computed':
    r"""
    Return the values of the matrix as an ``(a, b, c, d, e, f)`` tuple.
    Keywords: matplotlib.transforms.BlendedAffine2D, to_values
    """
    output_var = input_var.to_values()

def matplotlib_transforms_BlendedAffine2D_transform(self: 'any') -> 'any_computed':
    r"""
    Apply this transformation on the given array of *values*.
    Parameters
    ----------
    values : array-like
    Keywords: matplotlib.transforms.BlendedAffine2D, transform
    """
    output_var = input_var.transform()

def matplotlib_transforms_BlendedAffine2D_transform_affine(self: 'any') -> 'any_computed':
    r"""
    Apply only the affine part of this transformation on the
    given array of values.
    ``transform(values)`` is always equivalent to
    ``transform_affine(transform_non_affine(values))``.
    Keywords: matplotlib.transforms.BlendedAffine2D, transform_affine
    """
    output_var = input_var.transform_affine()

def matplotlib_transforms_BlendedAffine2D_transform_angles(self: 'any') -> 'any_computed':
    r"""
    Transform a set of angles anchored at specific locations.
    Parameters
    ----------
    angles : (N,) array-like
    Keywords: matplotlib.transforms.BlendedAffine2D, transform_angles
    """
    output_var = input_var.transform_angles()

def matplotlib_transforms_BlendedAffine2D_transform_bbox(self: 'any') -> 'any_computed':
    r"""
    Transform the given bounding box.
    For smarter transforms including caching (a common requirement in
    Matplotlib), see `TransformedBbox`.
    Keywords: matplotlib.transforms.BlendedAffine2D, transform_bbox
    """
    output_var = input_var.transform_bbox()

def matplotlib_transforms_BlendedAffine2D_transform_non_affine(self: 'any') -> 'any_computed':
    r"""
    Apply only the non-affine part of this transformation.
    ``transform(values)`` is always equivalent to
    ``transform_affine(transform_non_affine(values))``.
    Keywords: matplotlib.transforms.BlendedAffine2D, transform_non_affine
    """
    output_var = input_var.transform_non_affine()

def matplotlib_transforms_BlendedAffine2D_transform_path(self: 'any') -> 'any_computed':
    r"""
    Apply the transform to `.Path` *path*, returning a new `.Path`.
    In some cases, this transform may insert curves into the path
    that began as line segments.
    Keywords: matplotlib.transforms.BlendedAffine2D, transform_path
    """
    output_var = input_var.transform_path()

def matplotlib_transforms_BlendedAffine2D_transform_path_affine(self: 'any') -> 'any_computed':
    r"""
    Apply the affine part of this transform to `.Path` *path*, returning a
    new `.Path`.
    ``transform_path(path)`` is equivalent to
    ``transform_path_affine(transform_path_non_affine(values))``.
    Keywords: matplotlib.transforms.BlendedAffine2D, transform_path_affine
    """
    output_var = input_var.transform_path_affine()

def matplotlib_transforms_BlendedAffine2D_transform_path_non_affine(self: 'any') -> 'any_computed':
    r"""
    Apply the non-affine part of this transform to `.Path` *path*,
    returning a new `.Path`.
    ``transform_path(path)`` is equivalent to
    ``transform_path_affine(transform_path_non_affine(values))``.
    Keywords: matplotlib.transforms.BlendedAffine2D, transform_path_non_affine
    """
    output_var = input_var.transform_path_non_affine()

def matplotlib_transforms_BlendedAffine2D_transform_point(self: 'any') -> 'any_computed':
    r"""
    Return a transformed point.
    This function is only kept for backcompatibility; the more general
    `.transform` method is capable of transforming both a list of points
    and a single point.
    Keywords: matplotlib.transforms.BlendedAffine2D, transform_point
    """
    output_var = input_var.transform_point()

def matplotlib_transforms_BlendedGenericTransform_contains_branch(self: 'any') -> 'any_computed':
    r"""
    Return whether the given transform is a sub-tree of this transform.
    This routine uses transform equality to identify sub-trees, therefore
    in many situations it is object id which will be used.
    Keywords: matplotlib.transforms.BlendedGenericTransform, contains_branch
    """
    output_var = input_var.contains_branch()

def matplotlib_transforms_BlendedGenericTransform_contains_branch_seperately(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.transforms.BlendedGenericTransform, contains_branch_seperately
    """
    output_var = input_var.contains_branch_seperately()

def matplotlib_transforms_BlendedGenericTransform_frozen(self: 'any') -> 'any_computed':
    r"""
    Return a frozen copy of this transform node.  The frozen copy will not
    be updated when its children change.  Useful for storing a previously
    known state of a transform where ``copy.deepcopy()`` might normally be
    used.
    Keywords: matplotlib.transforms.BlendedGenericTransform, frozen
    """
    output_var = input_var.frozen()

def matplotlib_transforms_BlendedGenericTransform_get_affine(self: 'any') -> 'any_computed':
    r"""
    Get the affine part of this transform.
    Keywords: matplotlib.transforms.BlendedGenericTransform, get_affine
    """
    output_var = input_var.get_affine()

def matplotlib_transforms_BlendedGenericTransform_get_matrix(self: 'any') -> 'any_computed':
    r"""
    Get the matrix for the affine part of this transform.
    Keywords: matplotlib.transforms.BlendedGenericTransform, get_matrix
    """
    output_var = input_var.get_matrix()

def matplotlib_transforms_BlendedGenericTransform_invalidate(self: 'any') -> 'any_computed':
    r"""
    Invalidate this `TransformNode` and triggers an invalidation of its
    ancestors.  Should be called any time the transform changes.
    Keywords: matplotlib.transforms.BlendedGenericTransform, invalidate
    """
    output_var = input_var.invalidate()

def matplotlib_transforms_BlendedGenericTransform_inverted(self: 'any') -> 'any_computed':
    r"""
    Return the corresponding inverse transformation.
    It holds ``x == self.inverted().transform(self.transform(x))``.
    The return value of this method should be treated as
    Keywords: matplotlib.transforms.BlendedGenericTransform, inverted
    """
    output_var = input_var.inverted()

def matplotlib_transforms_BlendedGenericTransform_set_children(self: 'any') -> 'any_computed':
    r"""
    Set the children of the transform, to let the invalidation
    system know which transforms can invalidate this transform.
    Should be called from the constructor of any transforms that
    depend on other transforms.
    Keywords: matplotlib.transforms.BlendedGenericTransform, set_children
    """
    output_var = input_var.set_children()

def matplotlib_transforms_BlendedGenericTransform_transform(self: 'any') -> 'any_computed':
    r"""
    Apply this transformation on the given array of *values*.
    Parameters
    ----------
    values : array-like
    Keywords: matplotlib.transforms.BlendedGenericTransform, transform
    """
    output_var = input_var.transform()

def matplotlib_transforms_BlendedGenericTransform_transform_affine(self: 'any') -> 'any_computed':
    r"""
    Apply only the affine part of this transformation on the
    given array of values.
    ``transform(values)`` is always equivalent to
    ``transform_affine(transform_non_affine(values))``.
    Keywords: matplotlib.transforms.BlendedGenericTransform, transform_affine
    """
    output_var = input_var.transform_affine()

def matplotlib_transforms_BlendedGenericTransform_transform_angles(self: 'any') -> 'any_computed':
    r"""
    Transform a set of angles anchored at specific locations.
    Parameters
    ----------
    angles : (N,) array-like
    Keywords: matplotlib.transforms.BlendedGenericTransform, transform_angles
    """
    output_var = input_var.transform_angles()

def matplotlib_transforms_BlendedGenericTransform_transform_bbox(self: 'any') -> 'any_computed':
    r"""
    Transform the given bounding box.
    For smarter transforms including caching (a common requirement in
    Matplotlib), see `TransformedBbox`.
    Keywords: matplotlib.transforms.BlendedGenericTransform, transform_bbox
    """
    output_var = input_var.transform_bbox()

def matplotlib_transforms_BlendedGenericTransform_transform_non_affine(self: 'any') -> 'any_computed':
    r"""
    Apply only the non-affine part of this transformation.
    ``transform(values)`` is always equivalent to
    ``transform_affine(transform_non_affine(values))``.
    Keywords: matplotlib.transforms.BlendedGenericTransform, transform_non_affine
    """
    output_var = input_var.transform_non_affine()

def matplotlib_transforms_BlendedGenericTransform_transform_path(self: 'any') -> 'any_computed':
    r"""
    Apply the transform to `.Path` *path*, returning a new `.Path`.
    In some cases, this transform may insert curves into the path
    that began as line segments.
    Keywords: matplotlib.transforms.BlendedGenericTransform, transform_path
    """
    output_var = input_var.transform_path()

def matplotlib_transforms_BlendedGenericTransform_transform_path_affine(self: 'any') -> 'any_computed':
    r"""
    Apply the affine part of this transform to `.Path` *path*, returning a
    new `.Path`.
    ``transform_path(path)`` is equivalent to
    ``transform_path_affine(transform_path_non_affine(values))``.
    Keywords: matplotlib.transforms.BlendedGenericTransform, transform_path_affine
    """
    output_var = input_var.transform_path_affine()

def matplotlib_transforms_BlendedGenericTransform_transform_path_non_affine(self: 'any') -> 'any_computed':
    r"""
    Apply the non-affine part of this transform to `.Path` *path*,
    returning a new `.Path`.
    ``transform_path(path)`` is equivalent to
    ``transform_path_affine(transform_path_non_affine(values))``.
    Keywords: matplotlib.transforms.BlendedGenericTransform, transform_path_non_affine
    """
    output_var = input_var.transform_path_non_affine()

def matplotlib_transforms_BlendedGenericTransform_transform_point(self: 'any') -> 'any_computed':
    r"""
    Return a transformed point.
    This function is only kept for backcompatibility; the more general
    `.transform` method is capable of transforming both a list of points
    and a single point.
    Keywords: matplotlib.transforms.BlendedGenericTransform, transform_point
    """
    output_var = input_var.transform_point()

def matplotlib_transforms_CompositeAffine2D_contains_branch(self: 'any') -> 'any_computed':
    r"""
    Return whether the given transform is a sub-tree of this transform.
    This routine uses transform equality to identify sub-trees, therefore
    in many situations it is object id which will be used.
    Keywords: matplotlib.transforms.CompositeAffine2D, contains_branch
    """
    output_var = input_var.contains_branch()

def matplotlib_transforms_CompositeAffine2D_contains_branch_seperately(self: 'any') -> 'any_computed':
    r"""
    Return whether the given branch is a sub-tree of this transform on
    each separate dimension.
    A common use for this method is to identify if a transform is a blended
    transform containing an Axes' data transform. e.g.::
    Keywords: matplotlib.transforms.CompositeAffine2D, contains_branch_seperately
    """
    output_var = input_var.contains_branch_seperately()

def matplotlib_transforms_CompositeAffine2D_frozen(self: 'any') -> 'any_computed':
    r"""
    Return a frozen copy of this transform node.  The frozen copy will not
    be updated when its children change.  Useful for storing a previously
    known state of a transform where ``copy.deepcopy()`` might normally be
    used.
    Keywords: matplotlib.transforms.CompositeAffine2D, frozen
    """
    output_var = input_var.frozen()

def matplotlib_transforms_CompositeAffine2D_get_affine(self: 'any') -> 'any_computed':
    r"""
    Get the affine part of this transform.
    Keywords: matplotlib.transforms.CompositeAffine2D, get_affine
    """
    output_var = input_var.get_affine()

def matplotlib_transforms_CompositeAffine2D_get_matrix(self: 'any') -> 'any_computed':
    r"""
    Get the matrix for the affine part of this transform.
    Keywords: matplotlib.transforms.CompositeAffine2D, get_matrix
    """
    output_var = input_var.get_matrix()

def matplotlib_transforms_CompositeAffine2D_invalidate(self: 'any') -> 'any_computed':
    r"""
    Invalidate this `TransformNode` and triggers an invalidation of its
    ancestors.  Should be called any time the transform changes.
    Keywords: matplotlib.transforms.CompositeAffine2D, invalidate
    """
    output_var = input_var.invalidate()

def matplotlib_transforms_CompositeAffine2D_inverted(self: 'any') -> 'any_computed':
    r"""
    Return the corresponding inverse transformation.
    It holds ``x == self.inverted().transform(self.transform(x))``.
    The return value of this method should be treated as
    Keywords: matplotlib.transforms.CompositeAffine2D, inverted
    """
    output_var = input_var.inverted()

def matplotlib_transforms_CompositeAffine2D_set_children(self: 'any') -> 'any_computed':
    r"""
    Set the children of the transform, to let the invalidation
    system know which transforms can invalidate this transform.
    Should be called from the constructor of any transforms that
    depend on other transforms.
    Keywords: matplotlib.transforms.CompositeAffine2D, set_children
    """
    output_var = input_var.set_children()

def matplotlib_transforms_CompositeAffine2D_to_values(self: 'any') -> 'any_computed':
    r"""
    Return the values of the matrix as an ``(a, b, c, d, e, f)`` tuple.
    Keywords: matplotlib.transforms.CompositeAffine2D, to_values
    """
    output_var = input_var.to_values()

def matplotlib_transforms_CompositeAffine2D_transform(self: 'any') -> 'any_computed':
    r"""
    Apply this transformation on the given array of *values*.
    Parameters
    ----------
    values : array-like
    Keywords: matplotlib.transforms.CompositeAffine2D, transform
    """
    output_var = input_var.transform()

def matplotlib_transforms_CompositeAffine2D_transform_affine(self: 'any') -> 'any_computed':
    r"""
    Apply only the affine part of this transformation on the
    given array of values.
    ``transform(values)`` is always equivalent to
    ``transform_affine(transform_non_affine(values))``.
    Keywords: matplotlib.transforms.CompositeAffine2D, transform_affine
    """
    output_var = input_var.transform_affine()

def matplotlib_transforms_CompositeAffine2D_transform_angles(self: 'any') -> 'any_computed':
    r"""
    Transform a set of angles anchored at specific locations.
    Parameters
    ----------
    angles : (N,) array-like
    Keywords: matplotlib.transforms.CompositeAffine2D, transform_angles
    """
    output_var = input_var.transform_angles()

def matplotlib_transforms_CompositeAffine2D_transform_bbox(self: 'any') -> 'any_computed':
    r"""
    Transform the given bounding box.
    For smarter transforms including caching (a common requirement in
    Matplotlib), see `TransformedBbox`.
    Keywords: matplotlib.transforms.CompositeAffine2D, transform_bbox
    """
    output_var = input_var.transform_bbox()

def matplotlib_transforms_CompositeAffine2D_transform_non_affine(self: 'any') -> 'any_computed':
    r"""
    Apply only the non-affine part of this transformation.
    ``transform(values)`` is always equivalent to
    ``transform_affine(transform_non_affine(values))``.
    Keywords: matplotlib.transforms.CompositeAffine2D, transform_non_affine
    """
    output_var = input_var.transform_non_affine()

def matplotlib_transforms_CompositeAffine2D_transform_path(self: 'any') -> 'any_computed':
    r"""
    Apply the transform to `.Path` *path*, returning a new `.Path`.
    In some cases, this transform may insert curves into the path
    that began as line segments.
    Keywords: matplotlib.transforms.CompositeAffine2D, transform_path
    """
    output_var = input_var.transform_path()

def matplotlib_transforms_CompositeAffine2D_transform_path_affine(self: 'any') -> 'any_computed':
    r"""
    Apply the affine part of this transform to `.Path` *path*, returning a
    new `.Path`.
    ``transform_path(path)`` is equivalent to
    ``transform_path_affine(transform_path_non_affine(values))``.
    Keywords: matplotlib.transforms.CompositeAffine2D, transform_path_affine
    """
    output_var = input_var.transform_path_affine()

def matplotlib_transforms_CompositeAffine2D_transform_path_non_affine(self: 'any') -> 'any_computed':
    r"""
    Apply the non-affine part of this transform to `.Path` *path*,
    returning a new `.Path`.
    ``transform_path(path)`` is equivalent to
    ``transform_path_affine(transform_path_non_affine(values))``.
    Keywords: matplotlib.transforms.CompositeAffine2D, transform_path_non_affine
    """
    output_var = input_var.transform_path_non_affine()

def matplotlib_transforms_CompositeAffine2D_transform_point(self: 'any') -> 'any_computed':
    r"""
    Return a transformed point.
    This function is only kept for backcompatibility; the more general
    `.transform` method is capable of transforming both a list of points
    and a single point.
    Keywords: matplotlib.transforms.CompositeAffine2D, transform_point
    """
    output_var = input_var.transform_point()

def matplotlib_transforms_CompositeGenericTransform_contains_branch(self: 'any') -> 'any_computed':
    r"""
    Return whether the given transform is a sub-tree of this transform.
    This routine uses transform equality to identify sub-trees, therefore
    in many situations it is object id which will be used.
    Keywords: matplotlib.transforms.CompositeGenericTransform, contains_branch
    """
    output_var = input_var.contains_branch()

def matplotlib_transforms_CompositeGenericTransform_contains_branch_seperately(self: 'any') -> 'any_computed':
    r"""
    Return whether the given branch is a sub-tree of this transform on
    each separate dimension.
    A common use for this method is to identify if a transform is a blended
    transform containing an Axes' data transform. e.g.::
    Keywords: matplotlib.transforms.CompositeGenericTransform, contains_branch_seperately
    """
    output_var = input_var.contains_branch_seperately()

def matplotlib_transforms_CompositeGenericTransform_frozen(self: 'any') -> 'any_computed':
    r"""
    Return a frozen copy of this transform node.  The frozen copy will not
    be updated when its children change.  Useful for storing a previously
    known state of a transform where ``copy.deepcopy()`` might normally be
    used.
    Keywords: matplotlib.transforms.CompositeGenericTransform, frozen
    """
    output_var = input_var.frozen()

def matplotlib_transforms_CompositeGenericTransform_get_affine(self: 'any') -> 'any_computed':
    r"""
    Get the affine part of this transform.
    Keywords: matplotlib.transforms.CompositeGenericTransform, get_affine
    """
    output_var = input_var.get_affine()

def matplotlib_transforms_CompositeGenericTransform_get_matrix(self: 'any') -> 'any_computed':
    r"""
    Get the matrix for the affine part of this transform.
    Keywords: matplotlib.transforms.CompositeGenericTransform, get_matrix
    """
    output_var = input_var.get_matrix()

def matplotlib_transforms_CompositeGenericTransform_invalidate(self: 'any') -> 'any_computed':
    r"""
    Invalidate this `TransformNode` and triggers an invalidation of its
    ancestors.  Should be called any time the transform changes.
    Keywords: matplotlib.transforms.CompositeGenericTransform, invalidate
    """
    output_var = input_var.invalidate()

def matplotlib_transforms_CompositeGenericTransform_inverted(self: 'any') -> 'any_computed':
    r"""
    Return the corresponding inverse transformation.
    It holds ``x == self.inverted().transform(self.transform(x))``.
    The return value of this method should be treated as
    Keywords: matplotlib.transforms.CompositeGenericTransform, inverted
    """
    output_var = input_var.inverted()

def matplotlib_transforms_CompositeGenericTransform_set_children(self: 'any') -> 'any_computed':
    r"""
    Set the children of the transform, to let the invalidation
    system know which transforms can invalidate this transform.
    Should be called from the constructor of any transforms that
    depend on other transforms.
    Keywords: matplotlib.transforms.CompositeGenericTransform, set_children
    """
    output_var = input_var.set_children()

def matplotlib_transforms_CompositeGenericTransform_transform(self: 'any') -> 'any_computed':
    r"""
    Apply this transformation on the given array of *values*.
    Parameters
    ----------
    values : array-like
    Keywords: matplotlib.transforms.CompositeGenericTransform, transform
    """
    output_var = input_var.transform()

def matplotlib_transforms_CompositeGenericTransform_transform_affine(self: 'any') -> 'any_computed':
    r"""
    Apply only the affine part of this transformation on the
    given array of values.
    ``transform(values)`` is always equivalent to
    ``transform_affine(transform_non_affine(values))``.
    Keywords: matplotlib.transforms.CompositeGenericTransform, transform_affine
    """
    output_var = input_var.transform_affine()

def matplotlib_transforms_CompositeGenericTransform_transform_angles(self: 'any') -> 'any_computed':
    r"""
    Transform a set of angles anchored at specific locations.
    Parameters
    ----------
    angles : (N,) array-like
    Keywords: matplotlib.transforms.CompositeGenericTransform, transform_angles
    """
    output_var = input_var.transform_angles()

def matplotlib_transforms_CompositeGenericTransform_transform_bbox(self: 'any') -> 'any_computed':
    r"""
    Transform the given bounding box.
    For smarter transforms including caching (a common requirement in
    Matplotlib), see `TransformedBbox`.
    Keywords: matplotlib.transforms.CompositeGenericTransform, transform_bbox
    """
    output_var = input_var.transform_bbox()

def matplotlib_transforms_CompositeGenericTransform_transform_non_affine(self: 'any') -> 'any_computed':
    r"""
    Apply only the non-affine part of this transformation.
    ``transform(values)`` is always equivalent to
    ``transform_affine(transform_non_affine(values))``.
    Keywords: matplotlib.transforms.CompositeGenericTransform, transform_non_affine
    """
    output_var = input_var.transform_non_affine()

def matplotlib_transforms_CompositeGenericTransform_transform_path(self: 'any') -> 'any_computed':
    r"""
    Apply the transform to `.Path` *path*, returning a new `.Path`.
    In some cases, this transform may insert curves into the path
    that began as line segments.
    Keywords: matplotlib.transforms.CompositeGenericTransform, transform_path
    """
    output_var = input_var.transform_path()

def matplotlib_transforms_CompositeGenericTransform_transform_path_affine(self: 'any') -> 'any_computed':
    r"""
    Apply the affine part of this transform to `.Path` *path*, returning a
    new `.Path`.
    ``transform_path(path)`` is equivalent to
    ``transform_path_affine(transform_path_non_affine(values))``.
    Keywords: matplotlib.transforms.CompositeGenericTransform, transform_path_affine
    """
    output_var = input_var.transform_path_affine()

def matplotlib_transforms_CompositeGenericTransform_transform_path_non_affine(self: 'any') -> 'any_computed':
    r"""
    Apply the non-affine part of this transform to `.Path` *path*,
    returning a new `.Path`.
    ``transform_path(path)`` is equivalent to
    ``transform_path_affine(transform_path_non_affine(values))``.
    Keywords: matplotlib.transforms.CompositeGenericTransform, transform_path_non_affine
    """
    output_var = input_var.transform_path_non_affine()

def matplotlib_transforms_CompositeGenericTransform_transform_point(self: 'any') -> 'any_computed':
    r"""
    Return a transformed point.
    This function is only kept for backcompatibility; the more general
    `.transform` method is capable of transforming both a list of points
    and a single point.
    Keywords: matplotlib.transforms.CompositeGenericTransform, transform_point
    """
    output_var = input_var.transform_point()

def matplotlib_transforms_LockableBbox_anchored(self: 'any') -> 'any_computed':
    r"""
    Return a copy of the `Bbox` anchored to *c* within *container*.
    Parameters
    ----------
    c : (float, float) or {'C', 'SW', 'S', 'SE', 'E', 'NE', ...}
    Keywords: matplotlib.transforms.LockableBbox, anchored
    """
    output_var = input_var.anchored()

def matplotlib_transforms_LockableBbox_contains(self: 'any') -> 'any_computed':
    r"""
    Return whether ``(x, y)`` is in the bounding box or on its edge.
    Keywords: matplotlib.transforms.LockableBbox, contains
    """
    output_var = input_var.contains()

def matplotlib_transforms_LockableBbox_containsx(self: 'any') -> 'any_computed':
    r"""
    Return whether *x* is in the closed (:attr:`x0`, :attr:`x1`) interval.
    Keywords: matplotlib.transforms.LockableBbox, containsx
    """
    output_var = input_var.containsx()

def matplotlib_transforms_LockableBbox_containsy(self: 'any') -> 'any_computed':
    r"""
    Return whether *y* is in the closed (:attr:`y0`, :attr:`y1`) interval.
    Keywords: matplotlib.transforms.LockableBbox, containsy
    """
    output_var = input_var.containsy()

def matplotlib_transforms_LockableBbox_corners(self: 'any') -> 'any_computed':
    r"""
    Return the corners of this rectangle as an array of points.
    Specifically, this returns the array
    ``[[x0, y0], [x0, y1], [x1, y0], [x1, y1]]``.
    Keywords: matplotlib.transforms.LockableBbox, corners
    """
    output_var = input_var.corners()

def matplotlib_transforms_LockableBbox_count_contains(self: 'any') -> 'any_computed':
    r"""
    Count the number of vertices contained in the `Bbox`.
    Any vertices with a non-finite x or y value are ignored.
    Parameters
    ----------
    Keywords: matplotlib.transforms.LockableBbox, count_contains
    """
    output_var = input_var.count_contains()

def matplotlib_transforms_LockableBbox_count_overlaps(self: 'any') -> 'any_computed':
    r"""
    Count the number of bounding boxes that overlap this one.
    Parameters
    ----------
    bboxes : sequence of `.BboxBase`
    Keywords: matplotlib.transforms.LockableBbox, count_overlaps
    """
    output_var = input_var.count_overlaps()

def matplotlib_transforms_LockableBbox_expanded(self: 'any') -> 'any_computed':
    r"""
    Construct a `Bbox` by expanding this one around its center by the
    factors *sw* and *sh*.
    Keywords: matplotlib.transforms.LockableBbox, expanded
    """
    output_var = input_var.expanded()

def matplotlib_transforms_LockableBbox_frozen(self: 'any') -> 'any_computed':
    r"""
    The base class for anything that participates in the transform tree
    and needs to invalidate its parents or be invalidated.  This includes
    classes that are not really transforms, such as bounding boxes, since some
    transforms depend on bounding boxes to compute their values.
    Keywords: matplotlib.transforms.LockableBbox, frozen
    """
    output_var = input_var.frozen()

def matplotlib_transforms_LockableBbox_fully_contains(self: 'any') -> 'any_computed':
    r"""
    Return whether ``x, y`` is in the bounding box, but not on its edge.
    Keywords: matplotlib.transforms.LockableBbox, fully_contains
    """
    output_var = input_var.fully_contains()

def matplotlib_transforms_LockableBbox_fully_containsx(self: 'any') -> 'any_computed':
    r"""
    Return whether *x* is in the open (:attr:`x0`, :attr:`x1`) interval.
    Keywords: matplotlib.transforms.LockableBbox, fully_containsx
    """
    output_var = input_var.fully_containsx()

def matplotlib_transforms_LockableBbox_fully_containsy(self: 'any') -> 'any_computed':
    r"""
    Return whether *y* is in the open (:attr:`y0`, :attr:`y1`) interval.
    Keywords: matplotlib.transforms.LockableBbox, fully_containsy
    """
    output_var = input_var.fully_containsy()

def matplotlib_transforms_LockableBbox_fully_overlaps(self: 'any') -> 'any_computed':
    r"""
    Return whether this bounding box overlaps with the other bounding box,
    not including the edges.
    Parameters
    ----------
    Keywords: matplotlib.transforms.LockableBbox, fully_overlaps
    """
    output_var = input_var.fully_overlaps()

def matplotlib_transforms_LockableBbox_get_points(self: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.transforms.LockableBbox, get_points
    """
    output_var = input_var.get_points()

def matplotlib_transforms_LockableBbox_intersection(bbox1: 'any') -> 'any_computed':
    r"""
    Return the intersection of *bbox1* and *bbox2* if they intersect, or
    None if they don't.
    Keywords: matplotlib.transforms.LockableBbox, intersection
    """
    output_var = input_var.intersection()

def matplotlib_transforms_LockableBbox_invalidate(self: 'any') -> 'any_computed':
    r"""
    Invalidate this `TransformNode` and triggers an invalidation of its
    ancestors.  Should be called any time the transform changes.
    Keywords: matplotlib.transforms.LockableBbox, invalidate
    """
    output_var = input_var.invalidate()

def matplotlib_transforms_LockableBbox_overlaps(self: 'any') -> 'any_computed':
    r"""
    Return whether this bounding box overlaps with the other bounding box.
    Parameters
    ----------
    other : `.BboxBase`
    Keywords: matplotlib.transforms.LockableBbox, overlaps
    """
    output_var = input_var.overlaps()

def matplotlib_transforms_LockableBbox_padded(self: 'any') -> 'any_computed':
    r"""
    Construct a `Bbox` by padding this one on all four sides.
    Parameters
    ----------
    w_pad : float
    Keywords: matplotlib.transforms.LockableBbox, padded
    """
    output_var = input_var.padded()

def matplotlib_transforms_LockableBbox_rotated(self: 'any') -> 'any_computed':
    r"""
    Return the axes-aligned bounding box that bounds the result of rotating
    this `Bbox` by an angle of *radians*.
    Keywords: matplotlib.transforms.LockableBbox, rotated
    """
    output_var = input_var.rotated()

def matplotlib_transforms_LockableBbox_set_children(self: 'any') -> 'any_computed':
    r"""
    Set the children of the transform, to let the invalidation
    system know which transforms can invalidate this transform.
    Should be called from the constructor of any transforms that
    depend on other transforms.
    Keywords: matplotlib.transforms.LockableBbox, set_children
    """
    output_var = input_var.set_children()

def matplotlib_transforms_LockableBbox_shrunk(self: 'any') -> 'any_computed':
    r"""
    Return a copy of the `Bbox`, shrunk by the factor *mx*
    in the *x* direction and the factor *my* in the *y* direction.
    The lower left corner of the box remains unchanged.  Normally
    *mx* and *my* will be less than 1, but this is not enforced.
    Keywords: matplotlib.transforms.LockableBbox, shrunk
    """
    output_var = input_var.shrunk()

def matplotlib_transforms_LockableBbox_shrunk_to_aspect(self: 'any') -> 'any_computed':
    r"""
    Return a copy of the `Bbox`, shrunk so that it is as
    large as it can be while having the desired aspect ratio,
    *box_aspect*.  If the box coordinates are relative (i.e.
    fractions of a larger box such as a figure) then the
    physical aspect ratio of that figure is specified with
    Keywords: matplotlib.transforms.LockableBbox, shrunk_to_aspect
    """
    output_var = input_var.shrunk_to_aspect()

def matplotlib_transforms_LockableBbox_splitx(self: 'any') -> 'any_computed':
    r"""
    Return a list of new `Bbox` objects formed by splitting the original
    one with vertical lines at fractional positions given by *args*.
    Keywords: matplotlib.transforms.LockableBbox, splitx
    """
    output_var = input_var.splitx()

def matplotlib_transforms_LockableBbox_splity(self: 'any') -> 'any_computed':
    r"""
    Return a list of new `Bbox` objects formed by splitting the original
    one with horizontal lines at fractional positions given by *args*.
    Keywords: matplotlib.transforms.LockableBbox, splity
    """
    output_var = input_var.splity()

def matplotlib_transforms_LockableBbox_transformed(self: 'any') -> 'any_computed':
    r"""
    Construct a `Bbox` by statically transforming this one by *transform*.
    Keywords: matplotlib.transforms.LockableBbox, transformed
    """
    output_var = input_var.transformed()

def matplotlib_transforms_LockableBbox_translated(self: 'any') -> 'any_computed':
    r"""
    Construct a `Bbox` by translating this one by *tx* and *ty*.
    Keywords: matplotlib.transforms.LockableBbox, translated
    """
    output_var = input_var.translated()

def matplotlib_transforms_LockableBbox_union(bboxes: 'any') -> 'any_computed':
    r"""
    Return a `Bbox` that contains all of the given *bboxes*.
    Keywords: matplotlib.transforms.LockableBbox, union
    """
    output_var = input_var.union()

def matplotlib_transforms_ScaledTranslation_contains_branch(self: 'any') -> 'any_computed':
    r"""
    Return whether the given transform is a sub-tree of this transform.
    This routine uses transform equality to identify sub-trees, therefore
    in many situations it is object id which will be used.
    Keywords: matplotlib.transforms.ScaledTranslation, contains_branch
    """
    output_var = input_var.contains_branch()

def matplotlib_transforms_ScaledTranslation_contains_branch_seperately(self: 'any') -> 'any_computed':
    r"""
    Return whether the given branch is a sub-tree of this transform on
    each separate dimension.
    A common use for this method is to identify if a transform is a blended
    transform containing an Axes' data transform. e.g.::
    Keywords: matplotlib.transforms.ScaledTranslation, contains_branch_seperately
    """
    output_var = input_var.contains_branch_seperately()

def matplotlib_transforms_ScaledTranslation_frozen(self: 'any') -> 'any_computed':
    r"""
    Return a frozen copy of this transform node.  The frozen copy will not
    be updated when its children change.  Useful for storing a previously
    known state of a transform where ``copy.deepcopy()`` might normally be
    used.
    Keywords: matplotlib.transforms.ScaledTranslation, frozen
    """
    output_var = input_var.frozen()

def matplotlib_transforms_ScaledTranslation_get_affine(self: 'any') -> 'any_computed':
    r"""
    Get the affine part of this transform.
    Keywords: matplotlib.transforms.ScaledTranslation, get_affine
    """
    output_var = input_var.get_affine()

def matplotlib_transforms_ScaledTranslation_get_matrix(self: 'any') -> 'any_computed':
    r"""
    Get the matrix for the affine part of this transform.
    Keywords: matplotlib.transforms.ScaledTranslation, get_matrix
    """
    output_var = input_var.get_matrix()

def matplotlib_transforms_ScaledTranslation_invalidate(self: 'any') -> 'any_computed':
    r"""
    Invalidate this `TransformNode` and triggers an invalidation of its
    ancestors.  Should be called any time the transform changes.
    Keywords: matplotlib.transforms.ScaledTranslation, invalidate
    """
    output_var = input_var.invalidate()

def matplotlib_transforms_ScaledTranslation_inverted(self: 'any') -> 'any_computed':
    r"""
    Return the corresponding inverse transformation.
    It holds ``x == self.inverted().transform(self.transform(x))``.
    The return value of this method should be treated as
    Keywords: matplotlib.transforms.ScaledTranslation, inverted
    """
    output_var = input_var.inverted()

def matplotlib_transforms_ScaledTranslation_set_children(self: 'any') -> 'any_computed':
    r"""
    Set the children of the transform, to let the invalidation
    system know which transforms can invalidate this transform.
    Should be called from the constructor of any transforms that
    depend on other transforms.
    Keywords: matplotlib.transforms.ScaledTranslation, set_children
    """
    output_var = input_var.set_children()

def matplotlib_transforms_ScaledTranslation_to_values(self: 'any') -> 'any_computed':
    r"""
    Return the values of the matrix as an ``(a, b, c, d, e, f)`` tuple.
    Keywords: matplotlib.transforms.ScaledTranslation, to_values
    """
    output_var = input_var.to_values()

def matplotlib_transforms_ScaledTranslation_transform(self: 'any') -> 'any_computed':
    r"""
    Apply this transformation on the given array of *values*.
    Parameters
    ----------
    values : array-like
    Keywords: matplotlib.transforms.ScaledTranslation, transform
    """
    output_var = input_var.transform()

def matplotlib_transforms_ScaledTranslation_transform_affine(self: 'any') -> 'any_computed':
    r"""
    Apply only the affine part of this transformation on the
    given array of values.
    ``transform(values)`` is always equivalent to
    ``transform_affine(transform_non_affine(values))``.
    Keywords: matplotlib.transforms.ScaledTranslation, transform_affine
    """
    output_var = input_var.transform_affine()

def matplotlib_transforms_ScaledTranslation_transform_angles(self: 'any') -> 'any_computed':
    r"""
    Transform a set of angles anchored at specific locations.
    Parameters
    ----------
    angles : (N,) array-like
    Keywords: matplotlib.transforms.ScaledTranslation, transform_angles
    """
    output_var = input_var.transform_angles()

def matplotlib_transforms_ScaledTranslation_transform_bbox(self: 'any') -> 'any_computed':
    r"""
    Transform the given bounding box.
    For smarter transforms including caching (a common requirement in
    Matplotlib), see `TransformedBbox`.
    Keywords: matplotlib.transforms.ScaledTranslation, transform_bbox
    """
    output_var = input_var.transform_bbox()

def matplotlib_transforms_ScaledTranslation_transform_non_affine(self: 'any') -> 'any_computed':
    r"""
    Apply only the non-affine part of this transformation.
    ``transform(values)`` is always equivalent to
    ``transform_affine(transform_non_affine(values))``.
    Keywords: matplotlib.transforms.ScaledTranslation, transform_non_affine
    """
    output_var = input_var.transform_non_affine()

def matplotlib_transforms_ScaledTranslation_transform_path(self: 'any') -> 'any_computed':
    r"""
    Apply the transform to `.Path` *path*, returning a new `.Path`.
    In some cases, this transform may insert curves into the path
    that began as line segments.
    Keywords: matplotlib.transforms.ScaledTranslation, transform_path
    """
    output_var = input_var.transform_path()

def matplotlib_transforms_ScaledTranslation_transform_path_affine(self: 'any') -> 'any_computed':
    r"""
    Apply the affine part of this transform to `.Path` *path*, returning a
    new `.Path`.
    ``transform_path(path)`` is equivalent to
    ``transform_path_affine(transform_path_non_affine(values))``.
    Keywords: matplotlib.transforms.ScaledTranslation, transform_path_affine
    """
    output_var = input_var.transform_path_affine()

def matplotlib_transforms_ScaledTranslation_transform_path_non_affine(self: 'any') -> 'any_computed':
    r"""
    Apply the non-affine part of this transform to `.Path` *path*,
    returning a new `.Path`.
    ``transform_path(path)`` is equivalent to
    ``transform_path_affine(transform_path_non_affine(values))``.
    Keywords: matplotlib.transforms.ScaledTranslation, transform_path_non_affine
    """
    output_var = input_var.transform_path_non_affine()

def matplotlib_transforms_ScaledTranslation_transform_point(self: 'any') -> 'any_computed':
    r"""
    Return a transformed point.
    This function is only kept for backcompatibility; the more general
    `.transform` method is capable of transforming both a list of points
    and a single point.
    Keywords: matplotlib.transforms.ScaledTranslation, transform_point
    """
    output_var = input_var.transform_point()

def matplotlib_transforms_TransformNode_frozen(self: 'any') -> 'any_computed':
    r"""
    Return a frozen copy of this transform node.  The frozen copy will not
    be updated when its children change.  Useful for storing a previously
    known state of a transform where ``copy.deepcopy()`` might normally be
    used.
    Keywords: matplotlib.transforms.TransformNode, frozen
    """
    output_var = input_var.frozen()

def matplotlib_transforms_TransformNode_invalidate(self: 'any') -> 'any_computed':
    r"""
    Invalidate this `TransformNode` and triggers an invalidation of its
    ancestors.  Should be called any time the transform changes.
    Keywords: matplotlib.transforms.TransformNode, invalidate
    """
    output_var = input_var.invalidate()

def matplotlib_transforms_TransformNode_set_children(self: 'any') -> 'any_computed':
    r"""
    Set the children of the transform, to let the invalidation
    system know which transforms can invalidate this transform.
    Should be called from the constructor of any transforms that
    depend on other transforms.
    Keywords: matplotlib.transforms.TransformNode, set_children
    """
    output_var = input_var.set_children()

def matplotlib_transforms_TransformWrapper_contains_branch(self: 'any') -> 'any_computed':
    r"""
    Return whether the given transform is a sub-tree of this transform.
    This routine uses transform equality to identify sub-trees, therefore
    in many situations it is object id which will be used.
    Keywords: matplotlib.transforms.TransformWrapper, contains_branch
    """
    output_var = input_var.contains_branch()

def matplotlib_transforms_TransformWrapper_contains_branch_seperately(self: 'any') -> 'any_computed':
    r"""
    Return whether the given branch is a sub-tree of this transform on
    each separate dimension.
    A common use for this method is to identify if a transform is a blended
    transform containing an Axes' data transform. e.g.::
    Keywords: matplotlib.transforms.TransformWrapper, contains_branch_seperately
    """
    output_var = input_var.contains_branch_seperately()

def matplotlib_transforms_TransformWrapper_frozen(self: 'any') -> 'any_computed':
    r"""
    Return a frozen copy of this transform node.  The frozen copy will not
    be updated when its children change.  Useful for storing a previously
    known state of a transform where ``copy.deepcopy()`` might normally be
    used.
    Keywords: matplotlib.transforms.TransformWrapper, frozen
    """
    output_var = input_var.frozen()

def matplotlib_transforms_TransformWrapper_get_affine(self: 'any') -> 'any_computed':
    r"""
    Get the affine part of this transform.
    Keywords: matplotlib.transforms.TransformWrapper, get_affine
    """
    output_var = input_var.get_affine()

def matplotlib_transforms_TransformWrapper_get_matrix(self: 'any') -> 'any_computed':
    r"""
    Get the matrix for the affine part of this transform.
    Keywords: matplotlib.transforms.TransformWrapper, get_matrix
    """
    output_var = input_var.get_matrix()

def matplotlib_transforms_TransformWrapper_invalidate(self: 'any') -> 'any_computed':
    r"""
    Invalidate this `TransformNode` and triggers an invalidation of its
    ancestors.  Should be called any time the transform changes.
    Keywords: matplotlib.transforms.TransformWrapper, invalidate
    """
    output_var = input_var.invalidate()

def matplotlib_transforms_TransformWrapper_inverted(self: 'any') -> 'any_computed':
    r"""
    Return the corresponding inverse transformation.
    It holds ``x == self.inverted().transform(self.transform(x))``.
    The return value of this method should be treated as
    Keywords: matplotlib.transforms.TransformWrapper, inverted
    """
    output_var = input_var.inverted()

def matplotlib_transforms_TransformWrapper_set(self: 'any') -> 'any_computed':
    r"""
    Replace the current child of this transform with another one.
    The new child must have the same number of input and output
    dimensions as the current child.
    Keywords: matplotlib.transforms.TransformWrapper, set
    """
    output_var = input_var.set()

def matplotlib_transforms_TransformWrapper_set_children(self: 'any') -> 'any_computed':
    r"""
    Set the children of the transform, to let the invalidation
    system know which transforms can invalidate this transform.
    Should be called from the constructor of any transforms that
    depend on other transforms.
    Keywords: matplotlib.transforms.TransformWrapper, set_children
    """
    output_var = input_var.set_children()

def matplotlib_transforms_TransformWrapper_transform(self: 'any') -> 'any_computed':
    r"""
    Apply this transformation on the given array of *values*.
    Parameters
    ----------
    values : array-like
    Keywords: matplotlib.transforms.TransformWrapper, transform
    """
    output_var = input_var.transform()

def matplotlib_transforms_TransformWrapper_transform_affine(self: 'any') -> 'any_computed':
    r"""
    Apply only the affine part of this transformation on the
    given array of values.
    ``transform(values)`` is always equivalent to
    ``transform_affine(transform_non_affine(values))``.
    Keywords: matplotlib.transforms.TransformWrapper, transform_affine
    """
    output_var = input_var.transform_affine()

def matplotlib_transforms_TransformWrapper_transform_angles(self: 'any') -> 'any_computed':
    r"""
    Transform a set of angles anchored at specific locations.
    Parameters
    ----------
    angles : (N,) array-like
    Keywords: matplotlib.transforms.TransformWrapper, transform_angles
    """
    output_var = input_var.transform_angles()

def matplotlib_transforms_TransformWrapper_transform_bbox(self: 'any') -> 'any_computed':
    r"""
    Transform the given bounding box.
    For smarter transforms including caching (a common requirement in
    Matplotlib), see `TransformedBbox`.
    Keywords: matplotlib.transforms.TransformWrapper, transform_bbox
    """
    output_var = input_var.transform_bbox()

def matplotlib_transforms_TransformWrapper_transform_non_affine(self: 'any') -> 'any_computed':
    r"""
    Apply only the non-affine part of this transformation.
    ``transform(values)`` is always equivalent to
    ``transform_affine(transform_non_affine(values))``.
    Keywords: matplotlib.transforms.TransformWrapper, transform_non_affine
    """
    output_var = input_var.transform_non_affine()

def matplotlib_transforms_TransformWrapper_transform_path(self: 'any') -> 'any_computed':
    r"""
    Apply the transform to `.Path` *path*, returning a new `.Path`.
    In some cases, this transform may insert curves into the path
    that began as line segments.
    Keywords: matplotlib.transforms.TransformWrapper, transform_path
    """
    output_var = input_var.transform_path()

def matplotlib_transforms_TransformWrapper_transform_path_affine(self: 'any') -> 'any_computed':
    r"""
    Apply the affine part of this transform to `.Path` *path*, returning a
    new `.Path`.
    ``transform_path(path)`` is equivalent to
    ``transform_path_affine(transform_path_non_affine(values))``.
    Keywords: matplotlib.transforms.TransformWrapper, transform_path_affine
    """
    output_var = input_var.transform_path_affine()

def matplotlib_transforms_TransformWrapper_transform_path_non_affine(self: 'any') -> 'any_computed':
    r"""
    Apply the non-affine part of this transform to `.Path` *path*,
    returning a new `.Path`.
    ``transform_path(path)`` is equivalent to
    ``transform_path_affine(transform_path_non_affine(values))``.
    Keywords: matplotlib.transforms.TransformWrapper, transform_path_non_affine
    """
    output_var = input_var.transform_path_non_affine()

def matplotlib_transforms_TransformWrapper_transform_point(self: 'any') -> 'any_computed':
    r"""
    Return a transformed point.
    This function is only kept for backcompatibility; the more general
    `.transform` method is capable of transforming both a list of points
    and a single point.
    Keywords: matplotlib.transforms.TransformWrapper, transform_point
    """
    output_var = input_var.transform_point()

def matplotlib_transforms_blended_transform_factory(x_transform: 'any') -> 'any_computed':
    r"""
    Create a new "blended" transform using *x_transform* to transform
    the *x*-axis and *y_transform* to transform the *y*-axis.
    A faster version of the blended transform is returned for the case
    where both child transforms are affine.
    Keywords: matplotlib.transforms, blended_transform_factory
    """
    output_var = matplotlib.transforms.blended_transform_factory(input_var)

def matplotlib_transforms_composite_transform_factory(a: 'any') -> 'any_computed':
    r"""
    Create a new composite transform that is the result of applying
    transform a then transform b.
    Shortcut versions of the blended transform are provided for the
    case where both child transforms are affine, or one or the other
    Keywords: matplotlib.transforms, composite_transform_factory
    """
    output_var = matplotlib.transforms.composite_transform_factory(input_var)

def matplotlib_transforms_interval_contains(interval: 'any') -> 'any_computed':
    r"""
    Check, inclusively, whether an interval includes a given value.
    Parameters
    ----------
    interval : (float, float)
    Keywords: matplotlib.transforms, interval_contains
    """
    output_var = matplotlib.transforms.interval_contains(input_var)

def matplotlib_transforms_interval_contains_open(interval: 'any') -> 'any_computed':
    r"""
    Check, excluding endpoints, whether an interval includes a given value.
    Parameters
    ----------
    interval : (float, float)
    Keywords: matplotlib.transforms, interval_contains_open
    """
    output_var = matplotlib.transforms.interval_contains_open(input_var)

def matplotlib_transforms_inv(a: 'any') -> 'any_computed':
    r"""
    Compute the inverse of a matrix.
    Given a square matrix `a`, return the matrix `ainv` satisfying
    ``a @ ainv = ainv @ a = eye(a.shape[0])``.
    Keywords: matplotlib.transforms, inv
    """
    output_var = matplotlib.transforms.inv(input_var)

def matplotlib_transforms_nonsingular(vmin: 'any') -> 'any_computed':
    r"""
    Modify the endpoints of a range as needed to avoid singularities.
    Parameters
    ----------
    vmin, vmax : float
    Keywords: matplotlib.transforms, nonsingular
    """
    output_var = matplotlib.transforms.nonsingular(input_var)

def matplotlib_transforms_offset_copy(trans: 'any') -> 'any_computed':
    r"""
    Return a new transform with an added offset.
    Parameters
    ----------
    trans : `Transform` subclass
    Keywords: matplotlib.transforms, offset_copy
    """
    output_var = matplotlib.transforms.offset_copy(input_var)

def matplotlib_ticker_scale_range(vmin: 'any') -> 'any_computed':
    r"""
    Keywords: matplotlib.ticker, scale_range
    """
    output_var = matplotlib.ticker.scale_range(input_var)

def matplotlib_use(backend: 'any') -> 'any_computed':
    r"""
    Select the backend used for rendering and GUI integration.
    If pyplot is already imported, `~matplotlib.pyplot.switch_backend` is used
    and if the new backend is different than the current backend, all Figures
    will be closed.
    Keywords: matplotlib, use
    """
    output_var = matplotlib.use(input_var)

def matplotlib_validate_backend(s: 'any') -> 'any_computed':
    r"""
    [*Deprecated*] 
    Notes
    -----
    .. deprecated:: 3.10
    Keywords: matplotlib, validate_backend
    """
    output_var = matplotlib.validate_backend(input_var)


# Auto-synthesized EXPANSION nodes for matplotlib
import typing
import pandas as pd
import numpy as np
import cv2

def matplotlib_ExecutableNotFoundError___eq___default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.ExecutableNotFoundError.__eq__, matplotlib, ExecutableNotFoundError, equality, exception, comparison
    Node_Type: dunder
    """
    output_var = input_var.__eq__(value)

def matplotlib_ExecutableNotFoundError___format___default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.ExecutableNotFoundError.__format__, matplotlib, ExecutableNotFoundError, __format__
    Node_Type: dunder
    """
    output_var = input_var.__format__()

def matplotlib_ExecutableNotFoundError___ge___default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.ExecutableNotFoundError.__ge__, matplotlib, ExecutableNotFoundError, ge, comparison, operator
    Node_Type: dunder
    """
    output_var = input_var.__ge__()

def matplotlib_ExecutableNotFoundError___getstate___default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.ExecutableNotFoundError.__getstate__, matplotlib, ExecutableNotFoundError, __getstate__
    Node_Type: dunder
    """
    output_var = input_var.__getstate__()

def matplotlib_ExecutableNotFoundError___gt___default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.ExecutableNotFoundError.__gt__, matplotlib, ExecutableNotFoundError, greater than, __gt__, exception handling
    Node_Type: dunder
    """
    output_var = input_var.__gt__()

def matplotlib_ExecutableNotFoundError___hash___default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.ExecutableNotFoundError.__hash__, matplotlib, hash, ExecutableNotFoundError
    Node_Type: dunder
    """
    output_var = input_var.__hash__()

def matplotlib_ExecutableNotFoundError___init___default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.ExecutableNotFoundError.__init__, matplotlib, ExecutableNotFoundError, __init__
    Node_Type: dunder
    """
    output_var = input_var.__init__()

def matplotlib_ExecutableNotFoundError___init_subclass___default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.ExecutableNotFoundError.__init_subclass__, matplotlib, ExecutableNotFoundError, __init_subclass__
    Node_Type: dunder
    """
    output_var = input_var.__init_subclass__()

def matplotlib_ExecutableNotFoundError___le___default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.ExecutableNotFoundError.__le__, matplotlib, ExecutableNotFoundError, dunder, __le__
    Node_Type: dunder
    """
    output_var = input_var.__le__()

def matplotlib_ExecutableNotFoundError___lt___default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.ExecutableNotFoundError.__lt__, matplotlib, ExecutableNotFoundError, lt, comparison, operator
    Node_Type: dunder
    """
    output_var = input_var.__lt__()

def matplotlib_ExecutableNotFoundError___ne___default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.ExecutableNotFoundError.__ne__, matplotlib, ExecutableNotFoundError, ne, parameter, generator
    Node_Type: dunder
    """
    output_var = input_var.__ne__(value)

def matplotlib_ExecutableNotFoundError___reduce___default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.ExecutableNotFoundError.__reduce__, matplotlib, ExecutableNotFoundError, __reduce__
    Node_Type: dunder
    """
    output_var = input_var.__reduce__()

def matplotlib_ExecutableNotFoundError___reduce_ex___default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.ExecutableNotFoundError.__reduce_ex__, matplotlib, ExecutableNotFoundError, __reduce_ex__, protocol
    Node_Type: dunder
    """
    output_var = input_var.__reduce_ex__(4)

def matplotlib_ExecutableNotFoundError___repr___default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.ExecutableNotFoundError.__repr__, matplotlib, ExecutableNotFoundError, __repr__
    Node_Type: dunder
    """
    output_var = input_var.__repr__()

def matplotlib_ExecutableNotFoundError___setstate___default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.ExecutableNotFoundError.__setstate__, matplotlib, ExecutableNotFoundError, __setstate__, parameter, generator
    Node_Type: dunder
    """
    output_var = input_var.__setstate__()

def matplotlib_ExecutableNotFoundError___sizeof___default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.ExecutableNotFoundError.__sizeof__, matplotlib, ExecutableNotFoundError, sizeof
    Node_Type: dunder
    """
    output_var = input_var.__sizeof__()

def matplotlib_ExecutableNotFoundError___str___default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.ExecutableNotFoundError.__str__, matplotlib, ExecutableNotFoundError, __str__
    Node_Type: dunder
    """
    output_var = input_var.__str__()

def matplotlib_ExecutableNotFoundError_args_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.ExecutableNotFoundError.args, matplotlib, ExecutableNotFoundError, args
    Node_Type: property
    """
    output_var = input_var.args

def matplotlib_ExecutableNotFoundError_characters_written_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.ExecutableNotFoundError.characters_written, matplotlib, ExecutableNotFoundError, characters_written
    Node_Type: property
    """
    output_var = input_var.characters_written

def matplotlib_ExecutableNotFoundError_errno_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.ExecutableNotFoundError.errno, matplotlib, ExecutableNotFoundError, errno
    Node_Type: property
    """
    output_var = input_var.errno

def matplotlib_ExecutableNotFoundError_filename_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.ExecutableNotFoundError.filename, matplotlib, ExecutableNotFoundError, filename
    Node_Type: property
    """
    output_var = input_var.filename

def matplotlib_ExecutableNotFoundError_filename2_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.ExecutableNotFoundError.filename2, matplotlib, ExecutableNotFoundError, filename2
    Node_Type: property
    """
    output_var = input_var.filename2

def matplotlib_ExecutableNotFoundError_strerror_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.ExecutableNotFoundError.strerror, matplotlib, ExecutableNotFoundError, strerror
    Node_Type: property
    """
    output_var = input_var.strerror

def matplotlib_ExecutableNotFoundError_winerror_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.ExecutableNotFoundError.winerror, matplotlib, ExecutableNotFoundError, winerror
    Node_Type: property
    """
    output_var = input_var.winerror

def matplotlib_RcParams___class_getitem___default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.RcParams.__class_getitem__, matplotlib, RcParams, __class_getitem__
    Node_Type: dunder
    """
    output_var = input_var.__class_getitem__()

def matplotlib_RcParams___contains___default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.RcParams.__contains__, matplotlib, RcParams, contains, parameter, lookup
    Node_Type: dunder
    """
    output_var = input_var.__contains__(key)

def matplotlib_RcParams___delitem___default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.RcParams.__delitem__, matplotlib, RcParams, delete, parameter, configuration
    Node_Type: dunder
    """
    output_var = input_var.__delitem__(key)

def matplotlib_RcParams___getitem___default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.RcParams.__getitem__, matplotlib, RcParams, __getitem__, key
    Node_Type: dunder
    """
    output_var = input_var.__getitem__()

def matplotlib_RcParams___ior___default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.RcParams.__ior__, matplotlib, RcParams, ior, configuration
    Node_Type: dunder
    """
    output_var = input_var.__ior__(value)

def matplotlib_RcParams___iter___default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.RcParams.__iter__, matplotlib, RcParams, iteration
    Node_Type: dunder
    """
    output_var = input_var.__iter__()

def matplotlib_RcParams___len___default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.RcParams.__len__, matplotlib, RcParams, length, configuration, dunder method
    Node_Type: dunder
    """
    output_var = input_var.__len__()

def matplotlib_RcParams___or___default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.RcParams.__or__, matplotlib, RcParams, __or__
    Node_Type: dunder
    """
    output_var = input_var.__or__()

def matplotlib_RcParams___ror___default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.RcParams.__ror__, matplotlib, RcParams, __ror__
    Node_Type: dunder
    """
    output_var = input_var.__ror__(value)

def matplotlib_RcParams___setitem___default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.RcParams.__setitem__, matplotlib, RcParams, configuration, parameter, setitem
    Node_Type: dunder
    """
    output_var = input_var.__setitem__(key, val)

def matplotlib_artist_Artist_axes_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.artist.Artist.axes, matplotlib, artist, axes
    Node_Type: property
    """
    output_var = input_var.axes

def matplotlib_artist_Artist_figure_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.artist.Artist.figure, matplotlib, artist, figure
    Node_Type: property
    """
    output_var = input_var.figure

def matplotlib_artist_Artist_mouseover_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.artist.Artist.mouseover, matplotlib, mouseover, artist
    Node_Type: property
    """
    output_var = input_var.mouseover

def matplotlib_artist_Artist_stale_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.artist.Artist.stale, matplotlib, artist, stale
    Node_Type: property
    """
    output_var = input_var.stale

def matplotlib_artist_Artist_sticky_edges_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.artist.Artist.sticky_edges, matplotlib, artist, sticky_edges
    Node_Type: property
    """
    output_var = input_var.sticky_edges

def matplotlib_artist_Bbox___array___default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.artist.Bbox.__array__, matplotlib, bbox, array
    Node_Type: dunder
    """
    output_var = input_var.__array__()

def matplotlib_artist_Bbox___copy___default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.artist.Bbox.__copy__, matplotlib, artist, Bbox, copy
    Node_Type: dunder
    """
    output_var = input_var.__copy__()

def matplotlib_artist_Bbox_bounds_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.artist.Bbox.bounds, matplotlib, artist, Bbox, bounds
    Node_Type: property
    """
    output_var = input_var.bounds

def matplotlib_artist_Bbox_extents_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.artist.Bbox.extents, matplotlib, artist, Bbox, extents
    Node_Type: property
    """
    output_var = input_var.extents

def matplotlib_artist_Bbox_height_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.artist.Bbox.height, matplotlib, artist, Bbox, height, property
    Node_Type: property
    """
    output_var = input_var.height

def matplotlib_artist_Bbox_intervalx_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.artist.Bbox.intervalx, matplotlib, artist, Bbox, intervalx
    Node_Type: property
    """
    output_var = input_var.intervalx

def matplotlib_artist_Bbox_intervaly_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.artist.Bbox.intervaly, matplotlib, bbox, intervaly
    Node_Type: property
    """
    output_var = input_var.intervaly

def matplotlib_artist_Bbox_max_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.artist.Bbox.max, matplotlib, artist, Bbox, max
    Node_Type: property
    """
    output_var = input_var.max

def matplotlib_artist_Bbox_min_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.artist.Bbox.min, matplotlib, artist, Bbox, min
    Node_Type: property
    """
    output_var = input_var.min

def matplotlib_artist_Bbox_minpos_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.artist.Bbox.minpos, matplotlib, artist, Bbox, minpos
    Node_Type: property
    """
    output_var = input_var.minpos

def matplotlib_artist_Bbox_minposx_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.artist.Bbox.minposx, matplotlib, artist, Bbox, minposx, default value
    Node_Type: property
    """
    output_var = input_var.minposx

def matplotlib_artist_Bbox_minposy_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.artist.Bbox.minposy, matplotlib, artist, Bbox, minposy, default value
    Node_Type: property
    """
    output_var = input_var.minposy

def matplotlib_artist_Bbox_p0_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.artist.Bbox.p0, matplotlib, artist, Bbox, p0, default value
    Node_Type: property
    """
    output_var = input_var.p0

def matplotlib_artist_Bbox_p1_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.artist.Bbox.p1, matplotlib, artist, Bbox, p1, default value
    Node_Type: property
    """
    output_var = input_var.p1

def matplotlib_artist_Bbox_size_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.artist.Bbox.size, matplotlib, artist, Bbox, size
    Node_Type: property
    """
    output_var = input_var.size

def matplotlib_artist_Bbox_width_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.artist.Bbox.width, matplotlib, artist, Bbox, width
    Node_Type: property
    """
    output_var = input_var.width

def matplotlib_artist_Bbox_x0_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.artist.Bbox.x0, matplotlib, artist, Bbox, x0, default value
    Node_Type: property
    """
    output_var = input_var.x0

def matplotlib_artist_Bbox_x1_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.artist.Bbox.x1, matplotlib, artist, Bbox, x1, property
    Node_Type: property
    """
    output_var = input_var.x1

def matplotlib_artist_Bbox_xmax_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.artist.Bbox.xmax, matplotlib, artist, Bbox, xmax
    Node_Type: property
    """
    output_var = input_var.xmax

def matplotlib_artist_Bbox_xmin_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.artist.Bbox.xmin, matplotlib, artist, Bbox, xmin
    Node_Type: property
    """
    output_var = input_var.xmin

def matplotlib_artist_Bbox_y0_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.artist.Bbox.y0, matplotlib, artist, Bbox, y0, default value
    Node_Type: property
    """
    output_var = input_var.y0

def matplotlib_artist_Bbox_y1_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.artist.Bbox.y1, matplotlib, artist, Bbox, y1
    Node_Type: property
    """
    output_var = input_var.y1

def matplotlib_artist_Bbox_ymax_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.artist.Bbox.ymax, matplotlib, artist, Bbox, ymax
    Node_Type: property
    """
    output_var = input_var.ymax

def matplotlib_artist_Bbox_ymin_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.artist.Bbox.ymin, matplotlib, artist, Bbox, ymin, default value
    Node_Type: property
    """
    output_var = input_var.ymin

def matplotlib_artist_IdentityTransform___add___default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.artist.IdentityTransform.__add__, matplotlib, transform, addition, identity, other
    Node_Type: dunder
    """
    output_var = input_var.__add__(other)

def matplotlib_artist_IdentityTransform___sub___default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.artist.IdentityTransform.__sub__, matplotlib, transform, subtraction, identity, other
    Node_Type: dunder
    """
    output_var = input_var.__sub__(other)

def matplotlib_artist_IdentityTransform_depth_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.artist.IdentityTransform.depth, matplotlib, artist, IdentityTransform, depth
    Node_Type: property
    """
    output_var = input_var.depth

def matplotlib_artist_IdentityTransform_is_separable_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.artist.IdentityTransform.is_separable, matplotlib, artist, IdentityTransform, is_separable
    Node_Type: property
    """
    output_var = input_var.is_separable

def matplotlib_artist_Path___deepcopy___default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.artist.Path.__deepcopy__, matplotlib, deepcopy, Path, artist
    Node_Type: dunder
    """
    output_var = input_var.__deepcopy__()

def matplotlib_artist_Path_codes_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.artist.Path.codes, matplotlib, artist, Path, codes
    Node_Type: property
    """
    output_var = input_var.codes

def matplotlib_artist_Path_readonly_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.artist.Path.readonly, matplotlib, artist, Path, readonly
    Node_Type: property
    """
    output_var = input_var.readonly

def matplotlib_artist_Path_should_simplify_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.artist.Path.should_simplify, matplotlib, artist, Path, should_simplify
    Node_Type: property
    """
    output_var = input_var.should_simplify

def matplotlib_artist_Path_simplify_threshold_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.artist.Path.simplify_threshold, matplotlib, simplify_threshold, default, path, artist
    Node_Type: property
    """
    output_var = input_var.simplify_threshold

def matplotlib_artist_Path_vertices_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.artist.Path.vertices, matplotlib, artist, Path, vertices
    Node_Type: property
    """
    output_var = input_var.vertices

def matplotlib_cbook_silent_list___iadd___default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.cbook.silent_list.__iadd__, matplotlib, silent_list, iadd, parameter, dunder
    Node_Type: dunder
    """
    output_var = input_var.__iadd__(value)

def matplotlib_cbook_silent_list___imul___default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.cbook.silent_list.__imul__, matplotlib, silent_list, imul, multiplication, list
    Node_Type: dunder
    """
    output_var = input_var.__imul__(3)

def matplotlib_cbook_silent_list___mul___default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.cbook.silent_list.__mul__, matplotlib, silent_list, multiplication, dunder method, list multiplication
    Node_Type: dunder
    """
    output_var = input_var.__mul__(3)

def matplotlib_cbook_silent_list___reversed___default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.cbook.silent_list.__reversed__, matplotlib, cbook, silent_list, reversed
    Node_Type: dunder
    """
    output_var = input_var.__reversed__()

def matplotlib_cbook_silent_list___rmul___default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.cbook.silent_list.__rmul__, matplotlib, silent_list, rmul, multiplication, list
    Node_Type: dunder
    """
    output_var = input_var.__rmul__(3)

def matplotlib_backends_BackendFilter_name_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.backends.BackendFilter.name, matplotlib, backend, filter
    Node_Type: property
    """
    output_var = input_var.name

def matplotlib_backends_BackendFilter_value_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.backends.BackendFilter.value, matplotlib, backend, filter
    Node_Type: property
    """
    output_var = input_var.value

def matplotlib_bezier_BezierSegment___call___default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.bezier.BezierSegment.__call__, matplotlib, BezierSegment, call, parameter, t
    Node_Type: dunder
    """
    output_var = input_var.__call__(t)

def matplotlib_bezier_BezierSegment_control_points_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.bezier.BezierSegment.control_points, matplotlib, BezierSegment, control_points
    Node_Type: property
    """
    output_var = input_var.control_points

def matplotlib_bezier_BezierSegment_degree_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.bezier.BezierSegment.degree, matplotlib, BezierSegment, degree
    Node_Type: property
    """
    output_var = input_var.degree

def matplotlib_bezier_BezierSegment_dimension_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.bezier.BezierSegment.dimension, matplotlib, BezierSegment, dimension
    Node_Type: property
    """
    output_var = input_var.dimension

def matplotlib_bezier_BezierSegment_polynomial_coefficients_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.bezier.BezierSegment.polynomial_coefficients, matplotlib, BezierSegment, polynomial_coefficients
    Node_Type: property
    """
    output_var = input_var.polynomial_coefficients

def matplotlib_cm_ScalarMappable_cmap_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.cm.ScalarMappable.cmap, matplotlib, cmap, ScalarMappable
    Node_Type: property
    """
    output_var = input_var.cmap

def matplotlib_cm_ScalarMappable_colorbar_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.cm.ScalarMappable.colorbar, matplotlib, ScalarMappable, colorbar
    Node_Type: property
    """
    output_var = input_var.colorbar

def matplotlib_cm_ScalarMappable_norm_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.cm.ScalarMappable.norm, matplotlib, ScalarMappable, norm
    Node_Type: property
    """
    output_var = input_var.norm

def matplotlib_colors_AsinhNorm_clip_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.colors.AsinhNorm.clip, matplotlib, colors, AsinhNorm, clip
    Node_Type: property
    """
    output_var = input_var.clip

def matplotlib_colors_AsinhNorm_linear_width_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.colors.AsinhNorm.linear_width, matplotlib, colors, AsinhNorm, linear_width
    Node_Type: property
    """
    output_var = input_var.linear_width

def matplotlib_colors_AsinhNorm_vmax_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.colors.AsinhNorm.vmax, matplotlib, colors, AsinhNorm, vmax, default
    Node_Type: property
    """
    output_var = input_var.vmax

def matplotlib_colors_AsinhNorm_vmin_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.colors.AsinhNorm.vmin, matplotlib, colors, AsinhNorm, vmin
    Node_Type: property
    """
    output_var = input_var.vmin

def matplotlib_colors_BivarColormap_lut_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.colors.BivarColormap.lut, matplotlib, colormap, lut
    Node_Type: property
    """
    output_var = input_var.lut

def matplotlib_colors_BivarColormap_origin_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.colors.BivarColormap.origin, matplotlib, colormap, origin
    Node_Type: property
    """
    output_var = input_var.origin

def matplotlib_colors_BivarColormap_shape_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.colors.BivarColormap.shape, matplotlib, BivarColormap, shape
    Node_Type: property
    """
    output_var = input_var.shape

def matplotlib_colors_CenteredNorm_halfrange_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.colors.CenteredNorm.halfrange, matplotlib, colors, CenteredNorm, halfrange
    Node_Type: property
    """
    output_var = input_var.halfrange

def matplotlib_colors_CenteredNorm_vcenter_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.colors.CenteredNorm.vcenter, matplotlib, colors, CenteredNorm, vcenter
    Node_Type: property
    """
    output_var = input_var.vcenter

def matplotlib_colors_LightSource_direction_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.colors.LightSource.direction, matplotlib, LightSource, direction
    Node_Type: property
    """
    output_var = input_var.direction

def matplotlib_colors_MultivarColormap_combination_mode_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.colors.MultivarColormap.combination_mode, matplotlib, colormap, combination_mode
    Node_Type: property
    """
    output_var = input_var.combination_mode

def matplotlib_colors_SymLogNorm_linthresh_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.colors.SymLogNorm.linthresh, matplotlib, SymLogNorm, linthresh
    Node_Type: property
    """
    output_var = input_var.linthresh

def matplotlib_scale_FuncScaleLog_base_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.scale.FuncScaleLog.base, matplotlib, scale, log, base
    Node_Type: property
    """
    output_var = input_var.base

def matplotlib_scale_LogitLocator_minor_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.scale.LogitLocator.minor, matplotlib, LogitLocator, minor, default, configuration
    Node_Type: property
    """
    output_var = input_var.minor

def matplotlib_scale_ScalarFormatter_useLocale_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.scale.ScalarFormatter.useLocale, matplotlib, ScalarFormatter, useLocale
    Node_Type: property
    """
    output_var = input_var.useLocale

def matplotlib_scale_ScalarFormatter_useMathText_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.scale.ScalarFormatter.useMathText, matplotlib, ScalarFormatter, useMathText
    Node_Type: property
    """
    output_var = input_var.useMathText

def matplotlib_scale_ScalarFormatter_useOffset_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.scale.ScalarFormatter.useOffset, matplotlib, ScalarFormatter, useOffset, default, formatting
    Node_Type: property
    """
    output_var = input_var.useOffset

def matplotlib_scale_ScalarFormatter_usetex_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.scale.ScalarFormatter.usetex, matplotlib, ScalarFormatter, usetex
    Node_Type: property
    """
    output_var = input_var.usetex

def matplotlib_scale_SymmetricalLogScale_linscale_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.scale.SymmetricalLogScale.linscale, matplotlib, scale, SymmetricalLogScale, linscale, default
    Node_Type: property
    """
    output_var = input_var.linscale

def matplotlib_colorizer_ColorizingArtist_colorizer_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.colorizer.ColorizingArtist.colorizer, matplotlib, colorizer, ColorizingArtist
    Node_Type: property
    """
    output_var = input_var.colorizer

def matplotlib_ft2font_FT2Font___buffer___default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.ft2font.FT2Font.__buffer__, matplotlib, ft2font, FT2Font, __buffer__, flags
    Node_Type: dunder
    """
    output_var = input_var.__buffer__(flags)

def matplotlib_ft2font_FT2Font___release_buffer___default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.ft2font.FT2Font.__release_buffer__, matplotlib, ft2font, release_buffer
    Node_Type: dunder
    """
    output_var = input_var.__release_buffer__(buffer)

def matplotlib_ft2font_FT2Font_ascender_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.ft2font.FT2Font.ascender, matplotlib, ft2font, ascender
    Node_Type: property
    """
    output_var = input_var.ascender

def matplotlib_ft2font_FT2Font_bbox_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.ft2font.FT2Font.bbox, matplotlib, ft2font, bbox
    Node_Type: property
    """
    output_var = input_var.bbox

def matplotlib_ft2font_FT2Font_descender_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.ft2font.FT2Font.descender, matplotlib, ft2font, descender
    Node_Type: property
    """
    output_var = input_var.descender

def matplotlib_ft2font_FT2Font_face_flags_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.ft2font.FT2Font.face_flags, matplotlib, ft2font, face_flags
    Node_Type: property
    """
    output_var = input_var.face_flags

def matplotlib_ft2font_FT2Font_family_name_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.ft2font.FT2Font.family_name, matplotlib, font, family
    Node_Type: property
    """
    output_var = input_var.family_name

def matplotlib_ft2font_FT2Font_fname_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.ft2font.FT2Font.fname, matplotlib, ft2font, fname
    Node_Type: property
    """
    output_var = input_var.fname

def matplotlib_ft2font_FT2Font_max_advance_height_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.ft2font.FT2Font.max_advance_height, matplotlib, ft2font, max_advance_height
    Node_Type: property
    """
    output_var = input_var.max_advance_height

def matplotlib_ft2font_FT2Font_max_advance_width_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.ft2font.FT2Font.max_advance_width, matplotlib, ft2font, max_advance_width
    Node_Type: property
    """
    output_var = input_var.max_advance_width

def matplotlib_ft2font_FT2Font_num_charmaps_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.ft2font.FT2Font.num_charmaps, matplotlib, ft2font, num_charmaps
    Node_Type: property
    """
    output_var = input_var.num_charmaps

def matplotlib_ft2font_FT2Font_num_faces_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.ft2font.FT2Font.num_faces, matplotlib, ft2font, num_faces
    Node_Type: property
    """
    output_var = input_var.num_faces

def matplotlib_ft2font_FT2Font_num_fixed_sizes_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.ft2font.FT2Font.num_fixed_sizes, matplotlib, ft2font, num_fixed_sizes
    Node_Type: property
    """
    output_var = input_var.num_fixed_sizes

def matplotlib_ft2font_FT2Font_num_glyphs_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.ft2font.FT2Font.num_glyphs, matplotlib, ft2font, num_glyphs
    Node_Type: property
    """
    output_var = input_var.num_glyphs

def matplotlib_ft2font_FT2Font_num_named_instances_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.ft2font.FT2Font.num_named_instances, matplotlib, ft2font, num_named_instances
    Node_Type: property
    """
    output_var = input_var.num_named_instances

def matplotlib_ft2font_FT2Font_postscript_name_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.ft2font.FT2Font.postscript_name, matplotlib, ft2font, postscript_name
    Node_Type: property
    """
    output_var = input_var.postscript_name

def matplotlib_ft2font_FT2Font_scalable_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.ft2font.FT2Font.scalable, matplotlib, ft2font, FT2Font, scalable, property
    Node_Type: property
    """
    output_var = input_var.scalable

def matplotlib_ft2font_FT2Font_style_flags_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.ft2font.FT2Font.style_flags, matplotlib, ft2font, style_flags
    Node_Type: property
    """
    output_var = input_var.style_flags

def matplotlib_ft2font_FT2Font_style_name_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.ft2font.FT2Font.style_name, matplotlib, font, style
    Node_Type: property
    """
    output_var = input_var.style_name

def matplotlib_ft2font_FT2Font_underline_position_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.ft2font.FT2Font.underline_position, matplotlib, ft2font, underline_position
    Node_Type: property
    """
    output_var = input_var.underline_position

def matplotlib_ft2font_FT2Font_underline_thickness_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.ft2font.FT2Font.underline_thickness, matplotlib, ft2font, underline_thickness
    Node_Type: property
    """
    output_var = input_var.underline_thickness

def matplotlib_ft2font_FT2Font_units_per_EM_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.ft2font.FT2Font.units_per_EM, matplotlib, ft2font, units_per_EM, font metrics, text rendering
    Node_Type: property
    """
    output_var = input_var.units_per_EM

def matplotlib_ft2font_Glyph_horiAdvance_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.ft2font.Glyph.horiAdvance, matplotlib, ft2font, Glyph, horiAdvance
    Node_Type: property
    """
    output_var = input_var.horiAdvance

def matplotlib_ft2font_Glyph_horiBearingX_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.ft2font.Glyph.horiBearingX, matplotlib, ft2font, Glyph, horiBearingX
    Node_Type: property
    """
    output_var = input_var.horiBearingX

def matplotlib_ft2font_Glyph_horiBearingY_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.ft2font.Glyph.horiBearingY, matplotlib, ft2font, Glyph, horiBearingY, default value
    Node_Type: property
    """
    output_var = input_var.horiBearingY

def matplotlib_ft2font_Glyph_linearHoriAdvance_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.ft2font.Glyph.linearHoriAdvance, matplotlib, ft2font, Glyph, linearHoriAdvance
    Node_Type: property
    """
    output_var = input_var.linearHoriAdvance

def matplotlib_ft2font_Glyph_vertAdvance_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.ft2font.Glyph.vertAdvance, matplotlib, ft2font, Glyph, vertAdvance
    Node_Type: property
    """
    output_var = input_var.vertAdvance

def matplotlib_ft2font_Glyph_vertBearingX_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.ft2font.Glyph.vertBearingX, matplotlib, ft2font, Glyph, vertBearingX
    Node_Type: property
    """
    output_var = input_var.vertBearingX

def matplotlib_ft2font_Glyph_vertBearingY_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.ft2font.Glyph.vertBearingY, matplotlib, ft2font, Glyph, vertBearingY
    Node_Type: property
    """
    output_var = input_var.vertBearingY

def matplotlib_rcsetup_CapStyle___getnewargs___default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.rcsetup.CapStyle.__getnewargs__, matplotlib, rcsetup, CapStyle, __getnewargs__
    Node_Type: dunder
    """
    output_var = input_var.__getnewargs__()

def matplotlib_rcsetup_CapStyle___mod___default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.rcsetup.CapStyle.__mod__, matplotlib, rcsetup, CapStyle, __mod__
    Node_Type: dunder
    """
    output_var = input_var.__mod__(1)

def matplotlib_rcsetup_CapStyle___rmod___default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.rcsetup.CapStyle.__rmod__, matplotlib, rcsetup, CapStyle, __rmod__
    Node_Type: dunder
    """
    output_var = input_var.__rmod__(None)

def matplotlib_ticker_LinearLocator_numticks_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.ticker.LinearLocator.numticks, matplotlib, LinearLocator, numticks
    Node_Type: property
    """
    output_var = input_var.numticks

def matplotlib_ticker_PercentFormatter_symbol_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.ticker.PercentFormatter.symbol, matplotlib, PercentFormatter, symbol
    Node_Type: property
    """
    output_var = input_var.symbol

def matplotlib_transforms_BlendedGenericTransform_has_inverse_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.transforms.BlendedGenericTransform.has_inverse, matplotlib, transforms, BlendedGenericTransform, has_inverse, property
    Node_Type: property
    """
    output_var = input_var.has_inverse

def matplotlib_transforms_BlendedGenericTransform_is_affine_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.transforms.BlendedGenericTransform.is_affine, matplotlib, transforms, BlendedGenericTransform, is_affine, property
    Node_Type: property
    """
    output_var = input_var.is_affine

def matplotlib_transforms_DEBUG_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.transforms.DEBUG, matplotlib, transforms, DEBUG
    Node_Type: constant
    """
    output_var = matplotlib.transforms.DEBUG

def matplotlib_transforms_LockableBbox_locked_x0_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.transforms.LockableBbox.locked_x0, matplotlib, transforms, LockableBbox, locked_x0, property
    Node_Type: property
    """
    output_var = input_var.locked_x0

def matplotlib_transforms_LockableBbox_locked_x1_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.transforms.LockableBbox.locked_x1, matplotlib, transforms, LockableBbox, locked_x1, property
    Node_Type: property
    """
    output_var = input_var.locked_x1

def matplotlib_transforms_LockableBbox_locked_y0_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.transforms.LockableBbox.locked_y0, matplotlib, bbox, locked_y0
    Node_Type: property
    """
    output_var = input_var.locked_y0

def matplotlib_transforms_LockableBbox_locked_y1_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.transforms.LockableBbox.locked_y1, matplotlib, bbox, locked_y1
    Node_Type: property
    """
    output_var = input_var.locked_y1

def matplotlib_transforms_TransformWrapper_input_dims_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.transforms.TransformWrapper.input_dims, matplotlib, transforms, TransformWrapper, input_dims
    Node_Type: property
    """
    output_var = input_var.input_dims

def matplotlib_transforms_TransformWrapper_output_dims_default(input_var: 'any') -> 'any_computed':
    """
    Keywords: matplotlib.transforms.TransformWrapper.output_dims, matplotlib, transforms, TransformWrapper, output_dims
    Node_Type: property
    """
    output_var = input_var.output_dims

