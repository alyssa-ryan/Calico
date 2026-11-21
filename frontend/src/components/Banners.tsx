import footerSmall from "../assets/footer-960.webp";
import footerLarge from "../assets/footer-1944.webp";
import headerSmall from "../assets/header-960.webp";
import headerLarge from "../assets/header-1942.webp";

/**
 * The artwork's edges fade out, so in dark mode it melts into the page; in light mode
 * it sits on a full-width band of its own near-black, framing the page. It is
 * decorative (empty alt): the app's name is already in the navigation.
 */
function Artwork({ small, large, width }: { small: string; large: string; width: number }) {
  return (
    <div className="bg-artwork dark:bg-transparent">
      <img
        src={small}
        srcSet={`${small} 960w, ${large} ${width}w`}
        sizes="(min-width: 48rem) 48rem, 100vw"
        width={width}
        height={809}
        alt=""
        decoding="async"
        className="mx-auto h-auto w-full max-w-3xl mask-x-from-90% mask-x-to-100% mask-y-from-96% mask-y-to-100%"
      />
    </div>
  );
}

export function Masthead() {
  return (
    <div data-testid="masthead">
      <Artwork small={headerSmall} large={headerLarge} width={1942} />
    </div>
  );
}

export function Footer() {
  return (
    <footer className="mt-16">
      <Artwork small={footerSmall} large={footerLarge} width={1944} />
    </footer>
  );
}
